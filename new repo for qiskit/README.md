# Qiskit Fall Fest 2026 — Phase 1: PS-O1
## Weighted Max-Cut: QAOA Circuit Construction and SWAP Bottleneck Analysis

**Team:** Phoenix (Six Members)  
**Track 2:** Quantum Optimization (QAOA / QUBO)  
**Problem Statement:** PS-O1 (Max-Cut on Weighted Graphs)  
**Institution:** SRM University-AP | QFF 2026  
**Submission Portal:** [https://qffsrmap2026.com/learning/hackathon](https://qffsrmap2026.com/learning/hackathon)  

---

## 1. Executive Summary & Challenge Context

This repository is our reconstructed, reproducible submission for **Phase 1** of Qiskit Fall Fest 2026.

The scientific core of PS-O1 explores **algorithm-architecture co-design**:
1. Formulating the weighted Max-Cut problem into a cost Hamiltonian $\hat{C} = \sum_{(u,v)} w_{uv} \frac{I - Z_u Z_v}{2}$.
2. Independently determining the exact classical optimum via exhaustive brute-force evaluation.
3. Implementing the parameterised Quantum Approximate Optimization Algorithm (QAOA) ansatz for depths $p=1$ and $p=2$.
4. Optimizing variational parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ using classical gradient-free optimizers (COBYLA).
5. Mapping and compiling equivalent logical quantum circuits onto physical target architectures:
   - **Processor A:** 5-qubit linear chain baseline (`[(0,1), (1,2), (2,3), (3,4)]`)
   - **Processor B:** 9-qubit sparse heavy-hex contemporary reference (12 edges)
6. Analyzing compilation overhead: CX counts, circuit depth, and SWAP routing penalties.

---

## 2. Authoritative Repository Structure Compliance

Per Section 2 (Page 3) of the official **QFF 2026 Phase 1 Master Guide**, our repository strictly follows the required layout:

```
project/
├── README.md                      # Problem statement, setup, methodology, results
├── main.ipynb                     # End-to-end reproducible experiment notebook
├── main.py                        # Executable CLI end-to-end script
├── requirements.txt               # Pinned dependencies (qiskit, qiskit-aer, networkx, etc.)
├── data/
│   ├── raw/                       # Raw input data / G6 benchmark
│   └── processed/                 # Cleaned arrays or matrices
├── processors/
│   ├── processor_A.json           # Processor A coupling map & config (5q)
│   └── processor_B.json           # Processor B coupling map & config (9q)
├── src/
│   ├── __init__.py                # Package initializer
│   ├── problem.py                 # Problem formulation & cost Hamiltonian construction
│   ├── processors.py              # Topology graph loader & validation
│   ├── routing.py                 # Transpilation & layout routing routines
│   ├── metrics.py                 # Circuit metrics (depth, CX, SWAP, fidelity)
│   └── visualization.py           # Plotting & coupling graph visualizers
├── results/
│   ├── tables/                    # Exported CSV/JSON benchmark comparison tables
│   └── figures/                   # Exported comparison figures & topologies
├── report.md                      # Architectural analysis, trade-off interpretation, limitations
└── docs/                          # Specifications and technical handoffs
```

---

## 3. Benchmark Graphs & Classical Reference

### Benchmark G6 (`data/weighted_maxcut_g6.json`)
- **Nodes:** 6 vertices ($V = \{0, 1, 2, 3, 4, 5\}$)
- **Edges:** 9 edges with integer weights $w \in [1, 5]$
  - `(0,1): 3`, `(0,2): 2`, `(0,5): 4`, `(1,2): 5`, `(1,3): 2`, `(2,3): 3`, `(2,4): 1`, `(3,4): 4`, `(4,5): 5`
  - Total weight: 29
- **Exact Classical Optimum:** Cut weight **24.0**
  - Unique optimal partitions: $\{0, 2, 4\} \mid \{1, 3, 5\}$
  - Optimal bitstrings (vertex order): `010101` and `101010`

### Official Synthetic Generator
Per Page 4 of the Master Guide, the official problem setup generator is:
```python
import networkx as nx
G = nx.erdos_renyi_graph(n=6, p=0.5, seed=42)
```

---

## 4. Hardware Architectures: Processor A vs Processor B

Authoritative parameters derived directly from Page 2 of the official **QFF 2026 Master Guide**:

| Parameter | Processor A (Baseline Reference) | Processor B (Contemporary Reference) |
|---|---|---|
| **Physical Qubit Budget** | 5 Qubits (Nodes 0 to 4) | 9 Qubits (Nodes 0 to 8) |
| **Coupling Topology** | T-shaped branching tree: `[(0,1),(0,2),(0,3),(3,4)]` (hub at qubit 0) | Sparse heavy-hex inspired graph (12 edges): `[(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)]` |
| **Basis Gate Set** | `{rz, sx, x, cx}` | `{rz, sx, x, cx}` |
| **Synthetic Noise Model** | 1q depolarizing: 0.1%, 2q CX: 0.5%, Readout: 1.0% | Identical rates for controlled comparison |
| **Transpilation Baseline** | Optimization Level = 1, Seed = 42, Shots = 4096 | Optimization Level = 1, Seed = 42, Shots = 4096 |

### Architectural Capacity Finding
- **Processor A Capacity Constraint:** 5 physical qubits. A 6-logical-qubit benchmark (such as G6 or 6-node Erdös-Rényi) **cannot fit** without circuit cutting or qubit reuse. The system raises `CapacityError`. Disguising this physical limitation is scientifically incorrect.
- **Processor B Capacity:** 9 physical qubits. Readily fits 6 logical qubits, enabling full comparative analysis of SWAP routing overhead.

---

## 5. Quickstart & Reproducibility

```powershell
# 1. Activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run unit & integration test suite
python -m pytest -q

# 3. Execute end-to-end pipeline
python main.py

# 4. Run QAOA simulation on benchmark G6
python -m scripts.run_qaoa_g6 --p 1 --seed 7 --output results/tables/g6_p1.json
```
