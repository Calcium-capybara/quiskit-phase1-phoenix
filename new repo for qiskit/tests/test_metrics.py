"""Unit tests for Exploratory Connectivity Mismatch Score and Metrics module (Member 5)."""

import pytest

from src.metrics import (
    calculate_approximation_ratio,
    calculate_connectivity_mismatch_score,
    calculate_routing_overhead,
)


def test_approximation_ratio_calculation():
    assert calculate_approximation_ratio(24.0, 24.0) == 1.0
    assert calculate_approximation_ratio(18.0, 24.0) == 0.75
    assert calculate_approximation_ratio(10.0, 0.0) is None


def test_routing_overhead_calculation():
    res = calculate_routing_overhead(compiled_2q=24, logical_2q=9)
    # 9 logical RZZ -> 18 baseline CX.
    # 24 compiled CX -> 6 extra CX -> 2 estimated SWAPs.
    assert res["baseline_unrouted_cx"] == 18.0
    assert res["compiled_cx"] == 24.0
    assert res["routing_overhead_cx"] == 6.0
    assert res["estimated_swap_count"] == 2.0
    assert pytest.approx(res["routing_cx_ratio"], 0.001) == 24.0 / 18.0


def test_connectivity_mismatch_score_perfect_embedding():
    # If all logical edges map to direct physical connections, excess distance is 0 -> M = 0.
    logical_edges = [(0, 1, 3.0), (1, 2, 2.0)]
    physical_edges = [(10, 11), (11, 12)]
    mapping = {0: 10, 1: 11, 2: 12}

    m = calculate_connectivity_mismatch_score(logical_edges, physical_edges, mapping)
    assert m == 0.0


def test_connectivity_mismatch_score_manual_distance():
    # Linear physical chain 0-1-2-3
    # Logical edge (u=0, v=1) with weight 2 mapped to (0, 2): distance = 2 -> excess = 1
    # Logical edge (u=0, v=2) with weight 4 mapped to (0, 1): distance = 1 -> excess = 0
    # M = (2 * 1 + 4 * 0) / (2 + 4) = 2 / 6 = 1/3 ~ 0.3333
    logical_edges = [(0, 1, 2.0), (0, 2, 4.0)]
    physical_edges = [(0, 1), (1, 2), (2, 3)]
    mapping = {0: 0, 1: 2, 2: 1}

    m = calculate_connectivity_mismatch_score(logical_edges, physical_edges, mapping)
    assert pytest.approx(m, 0.001) == 1.0 / 3.0


def test_connectivity_mismatch_score_incomplete_mapping_returns_none():
    logical_edges = [(0, 1, 1.0)]
    physical_edges = [(0, 1)]
    mapping = {0: 0}  # Missing vertex 1

    m = calculate_connectivity_mismatch_score(logical_edges, physical_edges, mapping)
    assert m is None


def test_connectivity_mismatch_score_disconnected_graph_returns_none():
    logical_edges = [(0, 1, 1.0)]
    physical_edges = [(0, 2), (1, 3)]  # 0 and 1 are in disconnected components
    mapping = {0: 0, 1: 1}

    m = calculate_connectivity_mismatch_score(logical_edges, physical_edges, mapping, num_physical_nodes=4)
    assert m is None
