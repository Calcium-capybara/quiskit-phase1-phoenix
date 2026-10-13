"""Sample logical (noiseless) QAOA run on G6, the team's final benchmark graph.

G6 is team-selected, not organizer-provided. This is a logical
simulation only: no processor topology, routing, or noise is involved.
Prints a JSON summary; writes a file only if --output is given.

Usage (from the repository root):
    python -m scripts.run_qaoa_g6 --p 1 --seed 7
    python -m scripts.run_qaoa_g6 --p 2 --seed 7 --maxiter 300 --output results/tables/g6_p2.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import qiskit
import scipy

from src.problem import DEFAULT_GRAPH_PATH, load_graph
from src.qaoa_solver import QAOAConfig, run_qaoa

SUMMARY_FIELDS = (
    "p", "optimal_value", "expected_cut", "expected_approximation_ratio", "optimal_solution_probability",
    "best_cut_with_statevector_probability", "statevector_probability_threshold",
    "sampled_mean_cut", "sampled_approximation_ratio", "best_observed_cut", "best_observed_bitstring",
    "best_observed_is_exact_optimum", "sampled_optimal_frequency", "most_frequent_bitstring", "optimizer_success", "gammas", "betas",
    "shots", "seed", "bit_order",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH, help="graph JSON (default: G6)")
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)  # Challenge Kit setting for reported samples
    ap.add_argument("--output", type=Path, default=None, help="optional path for the full JSON result")
    args = ap.parse_args()

    graph = load_graph(args.graph)
    cfg = QAOAConfig(p=args.p, seed=args.seed, maxiter=args.maxiter, n_restarts=args.restarts, shots=args.shots)
    result = run_qaoa(graph, cfg).to_dict()
    best = result["optimizer"]["restarts"][result["optimizer"]["best_restart"]]
    summary = {
        "graph": graph.name,
        "graph_status": graph.source,
        **{k: result[k] for k in SUMMARY_FIELDS},
        "optimizer_message": best["message"],
        "optimizer_nfev": best["nfev"],
        "environment": {"python": platform.python_version(), "qiskit": qiskit.__version__, "scipy": scipy.__version__},
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "full_result": result}, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
