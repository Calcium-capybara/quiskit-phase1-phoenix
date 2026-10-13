"""Correctness tests for src/problem.py (Member 3).

All graphs here are SYNTHETIC test instances; expected values are hand-computed.
"""

import math

import networkx as nx
import numpy as np
import pytest

from src.problem import (
    WeightedGraph,
    all_cut_values,
    approximation_ratio,
    assignment_to_bitstring,
    assignment_to_index,
    bitstring_to_assignment,
    cut_edges,
    cut_value,
    cut_value_from_bitstring,
    cut_value_from_partition,
    index_to_assignment,
    load_graph,
    partition,
    partition_to_assignment,
    solve_exact,
    synthetic_unweighted_graph,
    synthetic_weighted_graph,
)

# Hand-computed cut table for synthetic_weighted_graph():
# edges (0,1,1) (1,2,2) (2,3,3) (0,3,4) (0,2,5); assignment x = (x0, x1, x2, x3)
WEIGHTED_TABLE = {
    (0, 0, 0, 0): 0,
    (0, 0, 0, 1): 7,   # (2,3)=3 + (0,3)=4
    (0, 0, 1, 0): 10,  # (1,2)=2 + (2,3)=3 + (0,2)=5
    (0, 0, 1, 1): 11,  # (1,2)=2 + (0,3)=4 + (0,2)=5
    (0, 1, 0, 0): 3,   # (0,1)=1 + (1,2)=2
    (0, 1, 0, 1): 10,  # 1 + 2 + 3 + 4
    (0, 1, 1, 0): 9,   # 1 + 3 + 5
    (0, 1, 1, 1): 10,  # 1 + 4 + 5
}


# -- construction & validation ----------------------------------------------


def test_edges_are_canonicalised_and_sorted():
    g = WeightedGraph.from_edges(3, [(2, 1, 0.5), (1, 0, 2)])
    assert g.edges == ((0, 1, 2.0), (1, 2, 0.5))
    assert g.num_edges == 2 and g.total_weight == 2.5


def test_missing_weight_defaults_to_one():
    assert WeightedGraph.from_edges(2, [(0, 1)]).edges == ((0, 1, 1.0),)


@pytest.mark.parametrize("edges", [[(0, 1), (1, 0)], [(0, 1, 1.0), (0, 1, 2.0)]])
def test_duplicate_and_reversed_edges_rejected(edges):
    with pytest.raises(ValueError, match="duplicate"):
        WeightedGraph.from_edges(2, edges)


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        WeightedGraph.from_edges(2, [(1, 1, 1.0)])


@pytest.mark.parametrize("bad", [(0, 3), (-1, 0)])
def test_out_of_range_vertex_rejected(bad):
    with pytest.raises(ValueError, match="out of range"):
        WeightedGraph.from_edges(3, [bad])


@pytest.mark.parametrize("bad", [(0.0, 1), ("0", 1), (True, 1)])
def test_non_int_vertex_rejected(bad):
    with pytest.raises(TypeError):
        WeightedGraph.from_edges(2, [bad])


@pytest.mark.parametrize("w", [-1.0, math.inf, math.nan])
def test_invalid_weights_rejected(w):
    with pytest.raises(ValueError):
        WeightedGraph.from_edges(2, [(0, 1, w)])


@pytest.mark.parametrize("w", ["heavy", None, True])
def test_non_numeric_weights_rejected(w):
    with pytest.raises(TypeError):
        WeightedGraph.from_edges(2, [(0, 1, w)])


@pytest.mark.parametrize("n", [0, -2])
def test_invalid_num_nodes(n):
    with pytest.raises(ValueError):
        WeightedGraph.from_edges(n, [])


def test_malformed_edge_rejected():
    with pytest.raises(ValueError, match="edge must be"):
        WeightedGraph.from_edges(3, [(0, 1, 2.0, 9)])


def test_dict_json_roundtrip(tmp_path):
    g = synthetic_weighted_graph()
    path = tmp_path / "g.json"
    g.to_json(path)
    g2 = WeightedGraph.from_json(path)
    assert g2 == g and g2.source == "synthetic"


def test_networkx_roundtrip_with_relabelling():
    nxg = nx.Graph()
    nxg.add_edge("b", "a", weight=2.0)
    nxg.add_edge("c", "b")  # no weight -> 1.0
    g = WeightedGraph.from_networkx(nxg)
    assert g.labels == ("a", "b", "c")
    assert g.edges == ((0, 1, 2.0), (1, 2, 1.0))
    assert WeightedGraph.from_networkx(g.to_networkx()).edges == g.edges


def test_networkx_directed_rejected():
    with pytest.raises(ValueError):
        WeightedGraph.from_networkx(nx.DiGraph([(0, 1)]))


def test_isolated_vertices_allowed():
    g = WeightedGraph.from_edges(3, [(0, 1, 1.0)])
    assert cut_value(g, (0, 1, 1)) == 1.0


# -- cut value ----------------------------------------------------------------


@pytest.mark.parametrize("x,expected", [((0, 0), 0.0), ((1, 1), 0.0), ((0, 1), 2.5), ((1, 0), 2.5)])
def test_single_edge_cut(x, expected):
    g = WeightedGraph.from_edges(2, [(0, 1, 2.5)])
    assert cut_value(g, x) == expected


@pytest.mark.parametrize("x,expected", WEIGHTED_TABLE.items())
def test_weighted_cut_table(x, expected):
    g = synthetic_weighted_graph()
    assert cut_value(g, x) == expected
    # complement symmetry
    assert cut_value(g, tuple(1 - b for b in x)) == expected


def test_weights_are_actually_used():
    """Same topology, different weights -> different cut value."""
    g1 = WeightedGraph.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)])
    g2 = WeightedGraph.from_edges(3, [(0, 1, 1.0), (1, 2, 7.0)])
    x = (0, 0, 1)  # cuts only edge (1,2)
    assert cut_value(g1, x) == 1.0
    assert cut_value(g2, x) == 7.0


def test_cut_edges_lists_crossing_edges():
    g = synthetic_weighted_graph()
    assert cut_edges(g, (0, 0, 1, 1)) == ((0, 2, 5.0), (0, 3, 4.0), (1, 2, 2.0))


@pytest.mark.parametrize(
    "bad,exc",
    # Non-int entries (float, bool) are TypeError; wrong length / out-of-range ints are ValueError.
    [((0, 1, 0), ValueError), ((0, 1, 2, 0), ValueError), ((0, 1, 0.5, 0), TypeError),
     ((True, 0, 0, 0), TypeError), ("0101", TypeError)],
)
def test_invalid_assignments_rejected(bad, exc):
    with pytest.raises(exc):
        cut_value(synthetic_weighted_graph(), bad)


def test_numpy_assignment_accepted():
    assert cut_value(synthetic_weighted_graph(), np.array([0, 0, 1, 1])) == 11.0


# -- strict cut_value validation (hardening) -------------------------------------
# G6 hand checks: (0,1,0,1,0,1) cuts (0,1)3+(0,5)4+(1,2)5+(2,3)3+(3,4)4+(4,5)5 = 24.


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_cut_value_valid_g6_assignment(container):
    assert cut_value(load_graph(), container([0, 1, 0, 1, 0, 1])) == 24.0


@pytest.mark.parametrize(
    "bad",
    [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [], [0, 1, 2, 1, 0, 1], [0, 1, -1, 1, 0, 1],
     np.array([[0, 1, 0], [1, 0, 1]])],
    ids=["too_short", "too_long", "empty", "two", "minus_one", "2d_array"],
)
def test_cut_value_rejects_bad_lengths_and_values(bad):
    with pytest.raises(ValueError):
        cut_value(load_graph(), bad)


@pytest.mark.parametrize(
    "bad",
    [[0, 1, 0, 1, 0, 0.5], [0, 1, 0, 1, 0, 1.0], [False, True, False, True, False, True],
     np.array([False, True, False, True, False, True]), np.array([0.0, 1, 0, 1, 0, 1]),
     [0, 1, 0, 1, 0, "1"], [0, 1, 0, 1, 0, None]],
    ids=["half", "float_one", "bools", "numpy_bools", "numpy_floats", "str_entry", "none_entry"],
)
def test_cut_value_rejects_non_int_entries(bad):
    with pytest.raises(TypeError):
        cut_value(load_graph(), bad)


@pytest.mark.parametrize(
    "bad",
    ["010101", b"\x00\x01\x00\x01\x00\x01", {0, 1}, {0: 0, 1: 1}, (b for b in [0, 1, 0, 1, 0, 1]), 5, None],
    ids=["str", "bytes", "set", "dict", "generator", "int", "none"],
)
def test_cut_value_rejects_malformed_containers(bad):
    with pytest.raises(TypeError):
        cut_value(load_graph(), bad)


# -- bitstrings -----------------------------------------------------------------


def test_bitstring_orders_are_mirror_images():
    assert bitstring_to_assignment("0001", 4, order="qiskit") == (1, 0, 0, 0)
    assert bitstring_to_assignment("0001", 4, order="vertex") == (0, 0, 0, 1)


def test_bitstring_roundtrip():
    x = (1, 0, 1, 1, 0)
    for order in ("qiskit", "vertex"):
        assert bitstring_to_assignment(assignment_to_bitstring(x, order=order), 5, order=order) == x


def test_reversed_bitstring_changes_weighted_cut():
    """Guard against bit-order bugs: the reversal is NOT the complement here."""
    g = synthetic_weighted_graph()
    # x = (0,0,0,1): cut 7. Qiskit string puts vertex 0 rightmost -> "1000".
    assert cut_value_from_bitstring(g, "1000", order="qiskit") == 7.0
    # Reading the same string in vertex order gives x=(1,0,0,0): cut 1+4+5=10.
    assert cut_value_from_bitstring(g, "1000", order="vertex") == 10.0


@pytest.mark.parametrize("bad", ["01", "01201", "0 101", "abcd"])
def test_invalid_bitstrings_rejected(bad):
    with pytest.raises(ValueError):
        bitstring_to_assignment(bad, 4, order="qiskit")


def test_bitstring_order_required_and_validated():
    with pytest.raises(TypeError):
        bitstring_to_assignment("0101", 4)  # order is keyword-only and required
    with pytest.raises(ValueError):
        bitstring_to_assignment("0101", 4, order="msb")


def test_index_assignment_roundtrip():
    assert assignment_to_index((1, 0, 1)) == 5
    assert index_to_assignment(5, 3) == (1, 0, 1)
    assert index_to_assignment(1, 4) == (1, 0, 0, 0)  # vertex 0 is the least-significant bit
    with pytest.raises(ValueError):
        index_to_assignment(8, 3)


def test_partition():
    assert partition((0, 1, 1, 0)) == ((0, 3), (1, 2))


def test_partition_roundtrip_and_cut():
    g = synthetic_weighted_graph()
    assert partition_to_assignment(4, [0, 1], [2, 3]) == (0, 0, 1, 1)
    assert partition_to_assignment(4, *partition((1, 0, 1, 1))) == (1, 0, 1, 1)
    assert cut_value_from_partition(g, {0, 1}, {2, 3}) == 11.0  # hand-computed optimum
    assert cut_value_from_partition(g, [0, 1, 2], [3]) == 7.0   # (2,3)=3 + (0,3)=4
    assert cut_value_from_partition(g, [0, 1, 2, 3], []) == 0.0


@pytest.mark.parametrize(
    "side0,side1,exc",
    [([0, 1], [1, 2, 3], ValueError),   # overlap
     ([0, 1], [2], ValueError),          # vertex 3 missing
     ([0, 0, 1], [2, 3], ValueError),    # repeated vertex
     ([0, 1], [2, 4], ValueError),       # out of range
     ([0, 1.0], [2, 3], TypeError)],     # non-int vertex
)
def test_invalid_partitions_rejected(side0, side1, exc):
    with pytest.raises(exc):
        partition_to_assignment(4, side0, side1)


# -- exact reference ---------------------------------------------------------------


def test_all_cut_values_matches_cut_value():
    g = synthetic_weighted_graph()
    values = all_cut_values(g)
    for k in range(16):
        assert values[k] == cut_value(g, index_to_assignment(k, 4))


def test_exact_weighted_optimum():
    sol = solve_exact(synthetic_weighted_graph())
    assert sol.optimal_value == 11.0
    assert set(sol.optimal_assignments) == {(0, 0, 1, 1), (1, 1, 0, 0)}
    assert sol.num_evaluated == 16
    assert set(sol.to_dict()["optimal_bitstrings_qiskit"]) == {"1100", "0011"}


def test_exact_unweighted_optimum():
    sol = solve_exact(synthetic_unweighted_graph())
    # Square 0-1-2-3 plus diagonal 0-2: best is {0,2}|{1,3}, cutting the 4 square edges.
    assert sol.optimal_value == 4.0
    assert set(sol.optimal_assignments) == {(0, 1, 0, 1), (1, 0, 1, 0)}


def test_exact_triangle_has_six_optima():
    g = WeightedGraph.from_edges(3, [(0, 1), (1, 2), (0, 2)])
    sol = solve_exact(g)
    assert sol.optimal_value == 2.0 and len(sol.optimal_assignments) == 6


def test_exact_matches_networkx_bruteforce_on_random_graphs():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(3, 8))
        edges = [(u, v, float(rng.integers(1, 10))) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.6]
        g = WeightedGraph.from_edges(n, edges)
        nxg = g.to_networkx()
        best = 0.0
        for k in range(2**n):
            side = {i for i in range(n) if (k >> i) & 1}
            best = max(best, nx.cut_size(nxg, side, weight="weight"))
        assert solve_exact(g).optimal_value == best


def test_exact_guard_against_large_graphs():
    g = WeightedGraph.from_edges(25, [(0, 1)])
    with pytest.raises(ValueError, match="max_nodes"):
        solve_exact(g)


def test_edgeless_graph_has_zero_optimum_and_undefined_ratio():
    sol = solve_exact(WeightedGraph.from_edges(3, []))
    assert sol.optimal_value == 0.0
    assert approximation_ratio(0.0, sol.optimal_value) is None


def test_approximation_ratio():
    assert approximation_ratio(8.25, 11.0) == pytest.approx(0.75)


# -- encoders/decoders validate entries (audit fix) ------------------------------


@pytest.mark.parametrize(
    "bad,exc",
    [([2, 1], ValueError), ([0, -1], ValueError), ([True, 0], TypeError), ([0.7, 1], TypeError),
     ({0, 1}, TypeError), ("01", TypeError), ([], ValueError)],
    ids=["two", "minus_one", "bool", "float", "set", "str", "empty"],
)
@pytest.mark.parametrize(
    "fn",
    [lambda a: assignment_to_bitstring(a, order="vertex"), lambda a: assignment_to_bitstring(a, order="qiskit"),
     assignment_to_index, partition],
    ids=["to_bitstring_vertex", "to_bitstring_qiskit", "to_index", "partition"],
)
def test_assignment_helpers_reject_invalid_entries(fn, bad, exc):
    with pytest.raises(exc):
        fn(bad)


def test_partition_no_longer_drops_vertices():
    """Before the fix, partition([2, 0, 1]) silently returned ((1,), (2,)) and lost vertex 0."""
    with pytest.raises(ValueError):
        partition([2, 0, 1])


@pytest.mark.parametrize("index,n,exc", [(True, 1, TypeError), (1.0, 2, TypeError), (0, 0, ValueError),
                                         (0, True, TypeError), (-1, 3, ValueError)])
def test_index_to_assignment_validation(index, n, exc):
    with pytest.raises(exc):
        index_to_assignment(index, n)


@pytest.mark.parametrize("n,exc", [(0, ValueError), (True, TypeError), (2.0, TypeError)])
def test_bitstring_to_assignment_validates_num_nodes(n, exc):
    with pytest.raises(exc):
        bitstring_to_assignment("", n, order="qiskit")


def test_both_bit_orders_on_g6_non_palindromic_key():
    """Vertex order: leftmost = vertex 0. Qiskit order: rightmost = vertex 0. Never interchangeable."""
    g = load_graph()
    x = (1, 1, 0, 0, 0, 0)  # vertices {0,1} on side 1: (0,2)2 + (0,5)4 + (1,2)5 + (1,3)2 = 13
    assert assignment_to_bitstring(x, order="vertex") == "110000"
    assert assignment_to_bitstring(x, order="qiskit") == "000011"
    assert cut_value_from_bitstring(g, "110000", order="vertex") == 13.0
    assert cut_value_from_bitstring(g, "000011", order="qiskit") == 13.0
    assert cut_value_from_bitstring(g, "110000", order="qiskit") == 9.0  # misread: vertices {4,5}
    assert assignment_to_index(x) == 3  # vertex 0 = least-significant bit
