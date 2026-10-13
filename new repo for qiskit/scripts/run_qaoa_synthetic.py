"""Member 3 development experiment: logical (noiseless) QAOA vs exact optimum.

SYNTHETIC GRAPHS ONLY -- these are NOT the team benchmark (G6; see
scripts/run_qaoa_g6.py). No processor topology is involved here (logical circuits only).

Usage (from the repository root):
    python -m scripts.run_qaoa_synthetic
"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import qiskit
import scipy

from src.problem import synthetic_unweighted_graph, synthetic_weighted_graph
from src.qaoa_solver import QAOAConfig, run_qaoa

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
DEPTHS = (1, 2, 3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, rows = [], []
    for graph in (synthetic_weighted_graph(), synthetic_unweighted_graph()):
        for p in DEPTHS:
            cfg = QAOAConfig(p=p, optimizer="COBYLA", maxiter=200, n_restarts=3, shots=2048, seed=1234)
            r = run_qaoa(graph, cfg)
            runs.append(r.to_dict())
            rows.append({
                "graph": graph.name,
                "source": graph.source,
                "p": p,
                "exact_optimum": r.optimal_value,
                "expected_cut": round(r.expected_cut, 4),
                "expected_ratio": round(r.expected_approximation_ratio, 4),
                "sampled_mean_cut": round(r.sampled_mean_cut, 4),
                "best_observed_cut": r.best_observed_cut,
                "best_observed_bitstring_qiskit": r.best_observed_bitstring,
                "P(optimal)": round(r.optimal_solution_probability, 4),
                "logical_depth": r.circuit["logical_depth"],
                "logical_rzz": r.circuit["logical_two_qubit_gates"],
                "nfev_best_restart": r.optimizer["restarts"][r.optimizer["best_restart"]]["nfev"],
                "runtime_s": round(r.runtime_seconds, 2),
            })
            print(rows[-1])

    payload = {
        "WARNING": "SYNTHETIC development graphs, not the official benchmark. Logical noiseless simulation only.",
        "environment": {"python": platform.python_version(), "qiskit": qiskit.__version__, "scipy": scipy.__version__},
        "runs": runs,
    }
    (OUT_DIR / "synthetic_logical_qaoa.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(OUT_DIR / "synthetic_logical_qaoa_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_DIR / 'synthetic_logical_qaoa.json'} and summary CSV")


if __name__ == "__main__":
    main()
