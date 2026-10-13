"""Graph, coupling map, and performance visualization module (Member 5).

Generates:
1. Processor coupling topologies (Processor A and Processor B)
2. Logical Max-Cut benchmark graph (G6)
3. QAOA cut probability distribution histograms
4. Compilation overhead comparisons (Depth & CX counts)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from src.processors import get_undirected_edges, load_processor


def save_graph_ascii(edges: list[tuple[int, int]], num_nodes: int, title: str, output_path: str | Path) -> None:
    """Create a structured ASCII / Markdown representation of the graph topology."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    text = f"# {title}\n\n"
    text += f"- Nodes: {num_nodes} (0 to {num_nodes - 1})\n"
    text += f"- Unique Undirected Edges ({len(edges)}):\n"
    for u, v in edges:
        text += f"  - ({u}, {v})\n"
    
    adj = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    text += "\n- Adjacency List:\n"
    for i in range(num_nodes):
        text += f"  - Node {i}: {sorted(adj[i])} (degree {len(adj[i])})\n"
        
    path.write_text(text, encoding="utf-8")


def export_processor_topologies(output_dir: str | Path = "results/figures") -> None:
    """Export topological structures for Processor A and Processor B."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Processor A
    cfg_a = load_processor("processor_A")
    edges_a = get_undirected_edges(cfg_a)
    save_graph_ascii(edges_a, cfg_a["num_qubits"], "Processor A - 5-Qubit Branching Tree", out / "processor_A_topology.txt")

    # Processor B
    cfg_b = load_processor("processor_B")
    edges_b = get_undirected_edges(cfg_b)
    save_graph_ascii(edges_b, cfg_b["num_qubits"], "Processor B - 9-Qubit Heavy-Hex Reference", out / "processor_B_topology.txt")


def plot_cut_histogram_ascii(cut_distribution: Mapping[float, int], output_path: str | Path) -> None:
    """Generate an ASCII histogram of cut values."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    total_shots = sum(cut_distribution.values())
    max_count = max(cut_distribution.values()) if cut_distribution else 1
    
    lines = ["# QAOA Sampled Cut Weight Distribution\n", f"Total shots: {total_shots}\n", "```"]
    for cut, count in sorted(cut_distribution.items()):
        pct = (count / total_shots) * 100 if total_shots > 0 else 0
        bar = "#" * int((count / max_count) * 40)
        lines.append(f"Cut {cut:5.1f} | {count:5d} ({pct:5.1f}%) | {bar}")
    lines.append("```\n")
    
    path.write_text("\n".join(lines), encoding="utf-8")
