"""Processor topology definitions and loader functions (Member 2).

Conforms to the Qiskit Fall Fest 2026 Phase 1 Master Guide:
- Processor A: 5-qubit linear chain [(0,1),(1,2),(2,3),(3,4)]
- Processor B: 9-qubit heavy-hex-inspired graph (12 unique edges)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSOR_DIR = PROJECT_ROOT / "processors"


def load_processor(name: str) -> dict[str, Any]:
    """Load and validate a processor JSON configuration.
    
    Args:
        name: Name of processor, e.g. 'processor_A', 'processor_B', or 'processor_D'.
        
    Returns:
        dict containing 'name', 'num_qubits', 'basis_gates', 'coupling_map',
        'synthetic_noise', and 'description'.
    """
    if name not in {"processor_A", "processor_B", "processor_D"}:
        raise ValueError(f"Unknown processor: {name}")

    path = PROCESSOR_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Processor configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("name") != name:
        raise ValueError(f"Unexpected processor name in {path}: expected {name}, got {config.get('name')}")

    n = config.get("num_qubits")
    edges = config.get("coupling_map")

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("num_qubits must be a positive integer")

    if not isinstance(edges, list):
        raise ValueError("coupling_map must be a list")

    seen = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"Invalid edge: {edge!r}")

        a, b = edge
        if any(not isinstance(q, int) or isinstance(q, bool) for q in edge):
            raise ValueError(f"Qubit indices must be integers: {edge!r}")

        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Out-of-range edge: {edge!r}")

        if a == b:
            raise ValueError(f"Self-loop: {edge!r}")

        if (a, b) in seen:
            raise ValueError(f"Duplicate directed edge: {edge!r}")

        seen.add((a, b))

    for a, b in seen:
        if (b, a) not in seen:
            raise ValueError(f"Missing reverse edge for {(a, b)}")

    return config


def get_undirected_edges(config: dict[str, Any]) -> list[tuple[int, int]]:
    """Return each physical connection once as sorted (u, v) with u < v."""
    return sorted({
        (min(a, b), max(a, b))
        for a, b in config["coupling_map"]
    })


def get_coupling_map(name_or_config: str | dict[str, Any]):
    """Construct a Qiskit CouplingMap from processor name or config dictionary."""
    from qiskit.transpiler import CouplingMap
    if isinstance(name_or_config, str):
        config = load_processor(name_or_config)
    else:
        config = name_or_config
    return CouplingMap(couplinglist=config["coupling_map"])
