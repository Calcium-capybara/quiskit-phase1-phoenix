"""Strict loading/validation of the team G6 graph (data/weighted_maxcut_g6.json).

G6 is TEAM-DEFINED, not organizer-provided. Invalid cases are built in memory
from a deep copy of the real file (or written to tmp_path); the real file is
never modified.
"""

import copy
import hashlib
import json

import pytest

from src.problem import (
    DEFAULT_GRAPH_PATH,
    WeightedGraph,
    load_graph,
    parse_graph_dict,
    solve_exact,
)

# Specification table (vertex order 0..5).
G6_EDGES = ((0, 1, 3.0), (0, 2, 2.0), (0, 5, 4.0), (1, 2, 5.0), (1, 3, 2.0),
            (2, 3, 3.0), (2, 4, 1.0), (3, 4, 4.0), (4, 5, 5.0))
G6_SHA256 = "b1bd12b00ec702d1bacb3de32bae01b1f7844b07ceb9769ec7da732e8eff64b6"


@pytest.fixture
def raw():
    return json.loads(DEFAULT_GRAPH_PATH.read_text(encoding="utf-8"))


def mutated(raw, fn):
    d = copy.deepcopy(raw)
    fn(d)
    return d


# -- the real graph -------------------------------------------------------------


def test_g6_file_unchanged():
    assert hashlib.sha256(DEFAULT_GRAPH_PATH.read_bytes()).hexdigest() == G6_SHA256


def test_g6_loads_as_specified():
    g = load_graph()
    assert isinstance(g, WeightedGraph)
    assert g.name == "G6" and "not an organizer-provided" in g.source
    assert g.num_nodes == 6 and g.num_edges == 9
    assert g.edges == G6_EDGES
    assert g.total_weight == 29.0


def test_g6_exact_optimum_matches_declared_metadata(raw):
    sol = solve_exact(load_graph())
    declared = raw["known_exact_optimum"]
    assert sol.optimal_value == declared["cut_weight"] == 24
    assert declared["total_edge_weight"] == 29
    vertex_order = {"".join(map(str, a)) for a in sol.optimal_assignments}
    assert vertex_order == set(declared["optimal_bitstrings_vertex_order"]) == {"010101", "101010"}
    # Complementary optima score identically.
    from src.problem import cut_value
    assert cut_value(load_graph(), (0, 1, 0, 1, 0, 1)) == cut_value(load_graph(), (1, 0, 1, 0, 1, 0)) == 24.0


# -- top-level structure ---------------------------------------------------------------


@pytest.mark.parametrize("field", ["num_vertices", "vertices", "edges"])
def test_missing_required_field(raw, field):
    with pytest.raises(ValueError, match=field):
        parse_graph_dict(mutated(raw, lambda d: d.pop(field)))


@pytest.mark.parametrize("data", [[], "graph", None, 6])
def test_top_level_must_be_object(data):
    with pytest.raises(TypeError, match="JSON object"):
        parse_graph_dict(data)


@pytest.mark.parametrize("value,exc", [("6", TypeError), (6.0, TypeError), (True, TypeError),
                                       (None, TypeError), (0, ValueError), (-1, ValueError)])
def test_invalid_num_vertices(raw, value, exc):
    with pytest.raises(exc, match="num_vertices"):
        parse_graph_dict(mutated(raw, lambda d: d.update(num_vertices=value)))


@pytest.mark.parametrize(
    "vertices,exc",
    [("012345", TypeError), ({"0": 0}, TypeError), ([0, 1, 2, 3, 4, True], TypeError),
     ([0, 1, 2, 3, 4, 5.0], TypeError), ([0, 1, 2, 3, 4], ValueError), ([0, 1, 2, 3, 4, 5, 6], ValueError),
     ([0, 1, 2, 3, 4, 4], ValueError), ([5, 4, 3, 2, 1, 0], ValueError), ([1, 2, 3, 4, 5, 6], ValueError)],
)
def test_invalid_vertices(raw, vertices, exc):
    with pytest.raises(exc, match="vertices"):
        parse_graph_dict(mutated(raw, lambda d: d.update(vertices=vertices)))


def test_declared_count_inconsistent_with_vertex_list(raw):
    with pytest.raises(ValueError, match="'vertices' has 6 entries but 'num_vertices' is 7"):
        parse_graph_dict(mutated(raw, lambda d: d.update(num_vertices=7)))


@pytest.mark.parametrize("edges,exc", [({"u": 0}, TypeError), ("edges", TypeError), ([], ValueError)])
def test_invalid_edges_container(raw, edges, exc):
    with pytest.raises(exc, match="edges"):
        parse_graph_dict(mutated(raw, lambda d: d.update(edges=edges)))


# -- edge records -----------------------------------------------------------------


def set_edge(raw, i, **kw):
    return mutated(raw, lambda d: d["edges"][i].update(kw))


@pytest.mark.parametrize("record", [[0, 1, 3], (0, 1, 3), "0-1", None])
def test_edge_record_must_be_object(raw, record):
    with pytest.raises(TypeError, match=r"edges\[2\]"):
        parse_graph_dict(mutated(raw, lambda d: d["edges"].__setitem__(2, record)))


@pytest.mark.parametrize("key", ["u", "v", "weight"])
def test_edge_missing_key(raw, key):
    with pytest.raises(ValueError, match=rf"edges\[4\].*missing \['{key}'\]"):
        parse_graph_dict(mutated(raw, lambda d: d["edges"][4].pop(key)))


def test_edge_unexpected_key(raw):
    with pytest.raises(ValueError, match=r"unexpected \['w'\]"):
        parse_graph_dict(set_edge(raw, 0, w=3))


@pytest.mark.parametrize("value,exc", [(True, TypeError), (1.0, TypeError), ("1", TypeError),
                                       (None, TypeError), (6, ValueError), (-1, ValueError)])
@pytest.mark.parametrize("key", ["u", "v"])
def test_invalid_endpoint(raw, key, value, exc):
    with pytest.raises(exc, match=rf"edges\[3\]\.{key}"):
        parse_graph_dict(set_edge(raw, 3, **{key: value}))


def test_self_loop(raw):
    with pytest.raises(ValueError, match=r"edges\[0\] is a self-loop on vertex 1"):
        parse_graph_dict(set_edge(raw, 0, u=1, v=1))


def test_duplicate_edge(raw):
    d = mutated(raw, lambda d: d["edges"].append({"u": 0, "v": 1, "weight": 2}))
    with pytest.raises(ValueError, match=r"edges\[9\] \(0, 1\) duplicates edges\[0\]"):
        parse_graph_dict(d)


def test_reversed_duplicate_edge(raw):
    d = mutated(raw, lambda d: d["edges"].append({"u": 5, "v": 4, "weight": 5}))
    with pytest.raises(ValueError, match=r"duplicates edges\[8\]"):
        parse_graph_dict(d)


@pytest.mark.parametrize(
    "weight,exc",
    [(True, TypeError), (False, TypeError), (3.0, TypeError), (2.5, TypeError), ("3", TypeError),
     (None, TypeError), (0, ValueError), (-2, ValueError), (6, ValueError)],
    ids=["true", "false", "float_int", "float", "str", "none", "zero", "negative", "above_range"],
)
def test_invalid_weight(raw, weight, exc):
    with pytest.raises(exc, match=r"edges\[1\]\.weight"):
        parse_graph_dict(set_edge(raw, 1, weight=weight))


def test_weight_must_be_positive_when_no_range_declared(raw):
    d = mutated(raw, lambda d: d.pop("weight_range"))
    assert parse_graph_dict(set_edge(d, 1, weight=99)).weight(0, 2) == 99.0
    with pytest.raises(ValueError, match=">= 1"):
        parse_graph_dict(set_edge(d, 1, weight=0))


@pytest.mark.parametrize("wr,exc", [([1], TypeError), ([1, 5.0], TypeError), ([True, 5], TypeError),
                                    ("1-5", TypeError), ([5, 1], ValueError), ([0, 5], ValueError)])
def test_invalid_weight_range(raw, wr, exc):
    with pytest.raises(exc, match="weight_range"):
        parse_graph_dict(mutated(raw, lambda d: d.update(weight_range=wr)))


def test_benchmark_id_must_be_string(raw):
    with pytest.raises(TypeError, match="benchmark_id"):
        parse_graph_dict(mutated(raw, lambda d: d.update(benchmark_id=6)))


# -- files ---------------------------------------------------------------------------


def test_malformed_json_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"num_vertices": 6, "edges": [', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_graph(p)


def test_json_array_file(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_graph(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "nope.json")


def test_valid_copy_roundtrips_through_file(tmp_path, raw):
    p = tmp_path / "copy.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert load_graph(p) == load_graph()
