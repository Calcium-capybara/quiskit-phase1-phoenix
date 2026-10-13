"""Metrics calculation, aggregation, and exploratory analysis (Member 5).

Provides tools to compute and format:
- Max-Cut objective metrics (cut weight, approximation ratio, optimal frequency)
- Compilation architecture metrics (depth overhead, CX overhead, SWAP overhead)
- Exploratory Connectivity Mismatch Score (Prompt 5B)
- Export to JSON and CSV formats
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx


def calculate_approximation_ratio(observed_cut: float, optimal_cut: float) -> float | None:
    """Compute approximation ratio alpha = observed_cut / optimal_cut."""
    if optimal_cut == 0.0:
        return None
    return float(observed_cut) / float(optimal_cut)


def calculate_routing_overhead(
    compiled_2q: int,
    logical_2q: int,
) -> dict[str, float]:
    """Calculate 2-qubit routing overhead and ratios.
    
    Each logical RZZ decomposes into 2 CX gates on hardware.
    Additional CX gates reflect routing (SWAP) overhead, where each SWAP contributes 3 CX gates.
    """
    baseline_cx = 2 * logical_2q
    extra_cx = max(0, compiled_2q - baseline_cx)
    estimated_swaps = extra_cx / 3.0
    ratio = (compiled_2q / baseline_cx) if baseline_cx > 0 else 1.0

    return {
        "logical_2q_gates": float(logical_2q),
        "baseline_unrouted_cx": float(baseline_cx),
        "compiled_cx": float(compiled_2q),
        "routing_overhead_cx": float(extra_cx),
        "estimated_swap_count": float(estimated_swaps),
        "routing_cx_ratio": float(ratio),
    }


def calculate_connectivity_mismatch_score(
    logical_edges: Sequence[tuple[int, int, float]],
    physical_edges: Sequence[tuple[int, int]],
    mapping: Mapping[int, int],
    *,
    num_physical_nodes: int | None = None,
) -> float | None:
    """Calculate the Exploratory Connectivity Mismatch Score M (Prompt 5B).
    
    Formula:
        M = (sum_{(i,j) in E} w_ij * [d(pi(i), pi(j)) - 1]) / (sum_{(i,j) in E} w_ij)
        
    Where:
        - E: set of logical graph edges (i, j) with non-negative weight w_ij.
        - pi(i): maps logical vertex i to physical qubit index pi(i).
        - d(u, v): shortest-path distance in the undirected physical coupling graph.
        
    Interpretation:
        The score measures the weighted average extra distance beyond direct physical
        adjacency across all logical interacting pairs.
        - M = 0.0: Perfect embedding where every logical edge maps to a direct physical edge.
        - M > 0.0: Positive routing penalty; higher values indicate greater required SWAP paths.
        
    Limitations & Constraints:
        - Exploratory metric only; does not replace actual compilation or measured SWAP counts.
        - Returns None if mapping is missing, incomplete, or if the physical graph is disconnected
          between any required mapped qubits.
    """
    if not logical_edges:
        return 0.0
    if not mapping:
        return None

    # Construct physical graph
    phys_g = nx.Graph()
    if num_physical_nodes is not None:
        phys_g.add_nodes_from(range(num_physical_nodes))
    for u, v in physical_edges:
        phys_g.add_edge(u, v)

    total_weight = 0.0
    weighted_excess_dist = 0.0

    for item in logical_edges:
        u, v = item[0], item[1]
        w = float(item[2]) if len(item) > 2 else 1.0

        if u not in mapping or v not in mapping:
            return None  # Incomplete mapping

        phys_u = mapping[u]
        phys_v = mapping[v]

        if not nx.has_path(phys_g, phys_u, phys_v):
            return None  # Disconnected physical graph between mapped qubits

        dist = nx.shortest_path_length(phys_g, source=phys_u, target=phys_v)
        excess = max(0, dist - 1)

        weighted_excess_dist += w * excess
        total_weight += w

    if total_weight == 0.0:
        return 0.0

    return weighted_excess_dist / total_weight


def export_comparison_table_csv(
    rows: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write benchmark records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save structured experiment results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
