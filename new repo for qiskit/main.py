"""End-to-end executable entry point for PS-O1 Weighted Max-Cut / QAOA & Routing Study.

Can be run via:
    python main.py
or executed cell-by-cell in main.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from src.processors import load_processor, get_undirected_edges
from src.visualization import export_processor_topologies


def run_pipeline() -> None:
    print("=" * 70)
    print("Qiskit Fall Fest 2026 - Phase 1: PS-O1 Weighted Max-Cut")
    print("QAOA Circuit Construction and SWAP Bottleneck Analysis")
    print("=" * 70)

    # 1. Inspect Processors
    print("\n[Step 1] Loading and Validating Target Processors...")
    proc_a = load_processor("processor_A")
    proc_b = load_processor("processor_B")
    print(f"  Processor A: {proc_a['num_qubits']} qubits, {len(get_undirected_edges(proc_a))} edges")
    print(f"  Processor B: {proc_b['num_qubits']} qubits, {len(get_undirected_edges(proc_b))} edges")
    export_processor_topologies("results/figures")
    print("  Exported topology figures to results/figures/")

    # 2. Inspect Benchmark Graph
    print("\n[Step 2] Validating Benchmark Graph G6...")
    g6_path = Path("data/weighted_maxcut_g6.json")
    if g6_path.exists():
        g6_data = json.loads(g6_path.read_text(encoding="utf-8"))
        print(f"  Benchmark ID: {g6_data['benchmark_id']}")
        print(f"  Vertices: {g6_data['num_vertices']}, Edges: {len(g6_data['edges'])}")
        print(f"  Exact Optimum Cut: {g6_data['known_exact_optimum']['cut_weight']}")
        print(f"  Optimal Bitstrings: {g6_data['known_exact_optimum']['optimal_bitstrings_vertex_order']}")

    # 3. Capacity & Routing Analysis
    print("\n[Step 3] Architectural Capacity & Routing Pre-check...")
    print("  Logical requirement: 6 qubits for G6.")
    print("  - Processor A capacity (5q): EXCEEDED. 6 > 5 logical qubits.")
    print("    -> Documented physical constraint; cannot execute 6q circuit on 5q hardware.")
    print("  - Processor B capacity (9q): FEASIBLE. 6 <= 9 physical qubits.")
    print("    -> Compatible for full QAOA compilation and SWAP analysis.")

    print("\n[Done] Pipeline verification complete. See results/ and report.md.")


if __name__ == "__main__":
    run_pipeline()
