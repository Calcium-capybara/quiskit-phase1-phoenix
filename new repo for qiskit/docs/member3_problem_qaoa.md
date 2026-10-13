# Member 3 — Weighted Max-Cut and QAOA (`src/problem.py`, `src/qaoa_solver.py`)

> **Benchmark.** The Challenge Kit requires a fixed six-vertex weighted benchmark graph for comparing Processors A, B and D but does not prescribe a specific graph.
> - **G6** (`data/weighted_maxcut_g6.json`) is the team's **final, team-selected benchmark** (confirmed 2026-10-13). It is not an organizer-provided graph.
>   It has 6 vertices, 9 edges, integer weights 1–5, total weight 29, and exact optimum 24 at `010101` / `101010` in vertex order.
>   Load it with `load_graph()`.
> - Use G6 unchanged for every comparable A/B/D experiment. Its edges, weights and vertex order are frozen, and the file's checksum is pinned by a test.
> - The `synthetic_*` graphs are **SYNTHETIC** development and test instances only.

## 1. Problem and objective

Undirected graph G = (V, E), V = {0, …, n−1}, non-negative finite weights w_uv.
An assignment x ∈ {0,1}^n puts vertex i on side x_i.

    C(x) = Σ_{(u,v) ∈ E} w_uv · [x_u ≠ x_v]

`WeightedGraph` validation enforces the following:
- Integer vertices in range.
- No self-loops.
- No duplicate edges, including reversed duplicates `(0,1)` / `(1,0)`, so no edge is double-counted.
- Finite, non-negative weights.

Edges are stored canonically as `(u, v, w)` with u < v, sorted.

**Loading G6 (strict).** `load_graph(path=DEFAULT_GRAPH_PATH)` reads the team schema and returns a `WeightedGraph`. `parse_graph_dict(data)` does the same for in-memory data. Nothing is repaired or defaulted. Messages name the offending field or `edges[i]`.
- **Raises `ValueError`:**
  - required fields (`num_vertices`, `vertices`, `edges`) missing;
  - `num_vertices` < 1, or `vertices` not exactly `[0..n−1]` in order (this includes a count mismatch);
  - `edges` empty;
  - an edge record without exactly the keys `u`/`v`/`weight`;
  - an endpoint out of range, a self-loop, or a duplicate or reversed-duplicate edge;
  - a weight outside `weight_range` (or < 1 when no range is given);
  - invalid JSON.
- **Raises `TypeError`:** a top level that isn't an object; or a `num_vertices`, vertex, endpoint or weight that isn't an `int`. **`bool` is rejected explicitly** (`True` is an `int` subclass), and so are floats like `3.0` and strings.

**Evaluating a cut.** The objective can be computed from three input forms:
- an assignment tuple, with `cut_value`. Strict contract:
  - accepts a list, tuple or 1-D NumPy integer array, indexed by vertex;
  - `TypeError` for strings or bytes, unordered or non-sequence inputs (set, dict, generator, scalar), and non-`int` entries (bool, float, str, None);
  - `ValueError` for a wrong length (including an empty assignment), a multi-dimensional array, or integers other than 0/1;
  - valid assignments score exactly as before;
  - the same entry checks apply to `assignment_to_bitstring`, `assignment_to_index` and `partition`. Before this check existed, `partition([2,0,1])` silently dropped vertex 0;
- a bitstring with an explicit order, with `cut_value_from_bitstring`;
- a partition `(side0, side1)`, with `cut_value_from_partition`. The two sides must be disjoint and must cover every vertex.

**Exact reference.** `solve_exact(graph)` enumerates all 2^n assignments (vectorised NumPy). It returns the optimum and *all* optimal assignments, including complementary pairs. It refuses n > 20 unless `max_nodes` is raised explicitly. The optimum is never hardcoded.

## 2. QAOA formulation

Cost Hamiltonian: **Ĉ = Σ_{(u,v)} w_uv (I − Z_u Z_v)/2**.
Z_u Z_v|x⟩ = ±|x⟩ (+ if x_u = x_v), so ⟨x|Ĉ|x⟩ = C(x), and ⟨ψ|Ĉ|ψ⟩ is the expected cut. Its diagonal equals `all_cut_values(graph)`, and this is tested.

State: |γ,β⟩ = Π_{l=1..p} U_B(β_l) U_C(γ_l) |+⟩^n

| Operator | Definition | Qiskit gates |
|---|---|---|
| U_C(γ) | exp(−iγĈ) | `rzz(−w_uv·γ)` on every edge (global phase e^{−iγw/2} dropped) |
| U_B(β) | exp(−iβ Σ_j X_j) | `rx(2β)` on every qubit |

The derivation uses Qiskit's definition RZZ(θ) = exp(−iθ/2 ZZ). Then exp(−iγw(I−ZZ)/2) = e^{−iγw/2}·exp(+iγw/2 ZZ) = phase · RZZ(−wγ).

Zero-weight edges emit no gate.

The tests check this against an independent `scipy.linalg.expm` construction and the closed-form single-edge result ⟨C⟩ = (w/2)(1 + sin 4β · sin wγ). Flipping the sign makes 4 tests fail (mutation-checked).

*Convention warning:* tutorials that use `rzz(2wγ')` are using a different parameterisation (γ = −2γ'). Convert any angles taken from them.

- **p**: the number of alternating cost/mixer layers.
- **γ_l**: the cost-layer angles.
- **β_l**: the mixer angles.

The circuit has 2p `Parameter`s, named `gamma[l]` and `beta[l]`.

## 3. Bit-order convention (single mapping used everywhere)

    vertex i → logical qubit q[i] → classical bit meas[i] → counts-key character key[n−1−i]

The **rightmost** character of a Qiskit counts key is vertex 0. Decode with
`bitstring_to_assignment(key, n, order="qiskit")`. The `order` argument is mandatory so it can never be omitted by accident.
Example (n=4): key `"1000"` → x = (0,0,0,1) → cut 7 on the synthetic weighted graph. Reading the key the wrong way round gives 10, and the tests catch that (mutation-checked: 5 failures).

**G6 caveat.** G6's optimal strings `010101` and `101010` are reversals *and* complements of each other, so they score 24 in either reading. They **cannot detect** a reversal bug. The G6 bit-order tests therefore use non-symmetric keys:
- `"000011"` → vertices {0,1} → cut 13; reading it reversed gives 9.
- `"000100"` → vertex {2} → cut 11; reading it reversed gives 9.

## 4. Optimisation and evaluation (experiment settings, not official requirements)

`QAOAConfig` defaults:

| Setting | Default |
|---|---|
| `p` | 1 |
| `optimizer` | `"COBYLA"` (any gradient-free `scipy.optimize.minimize` method) |
| `maxiter` | 200 |
| `n_restarts` | 3 |
| `objective` | `"statevector"` (exact expectation) or `"shots"` |
| `shots` | 4096 (Challenge Kit setting for reported sampled results) |
| `seed` | 1234 |
| `statevector_probability_threshold` | `1e-12` |

**Config validation.** `QAOAConfig.validate()` runs inside `run_qaoa` and `optimize_parameters`, before any work starts. Nothing is coerced or silently repaired.
- **`TypeError`:**
  - `p`, `maxiter`, `n_restarts` or `shots` that aren't `int` (including `bool`);
  - a `seed` that isn't an `int`;
  - an empty or non-string `optimizer`;
  - malformed ranges or a malformed `initial_point`.
- **`ValueError`:**
  - `p`, `maxiter`, `n_restarts` or `shots` below 1, or `seed` < 0;
  - `tol` that is ≤ 0 or non-finite;
  - ranges with non-finite values or low ≥ high;
  - a non-finite `initial_point` value or a length other than 2p;
  - a threshold that isn't finite or isn't in [0, 1).

Previously, `shots=True`, `maxiter=2.5`, `seed=True` and `tol=-1` were accepted silently. SciPy then silently replaced the invalid `tol`.

**Initialization ranges are not bounds.** Random starting angles are drawn from `gamma_range = [0, π)` and `beta_range = [0, π/2)` in `optimize_parameters`. `scipy.optimize.minimize(method="COBYLA")` is called **without `bounds` or constraints**, so the optimizer may move outside these ranges.
- The returned `gammas` and `betas` can lie outside them; an earlier synthetic run returned β = −0.39.
- A user-supplied `initial_point` is used as given, without clipping.
- A test checks that no bounds are passed.
- The β range covers every distinct mixer angle, because the expectation is π/2-periodic in β. The γ range is a heuristic, not a required QAOA range.

### Result schema (`run_qaoa(graph, config)` → `QAOAResult`, `.to_dict()` is JSON-safe)

| Kind | Fields | Meaning / guarantee |
|---|---|---|
| Exact reference (deterministic) | `exact`, `optimal_value` | Exhaustive-search optimum. `None` if n > `max_exact_nodes` |
| Optimized parameters | `gammas`, `betas`, `p` | Best restart's final angles (unbounded; see above) |
| Statevector (deterministic for given angles, up to floating point) | `expected_cut`, `expected_approximation_ratio` | Exact ⟨Ĉ⟩ and ⟨Ĉ⟩ / optimum |
| | `optimal_solution_probability` | Total probability on all optimal assignments |
| | `best_cut_with_statevector_probability`, `statevector_probability_threshold` | Max cut over basis states with probability **strictly >** threshold (default 1e-12; must be finite and in [0, 1)). `None` if no state qualifies. **Not** a sampled result, and **not** evidence of optimization: even the uniform state at γ = β = 0 supports 24 on G6 (a test checks this). Use `optimal_solution_probability` instead |
| Finite-shot (stochastic, seeded) | `counts`, `cut_distribution`, `sampled_mean_cut`, `sampled_approximation_ratio`, `shots`, `seed` | Noiseless `StatevectorSampler` samples (Qiskit-order keys) |
| | `best_observed_cut` / `_bitstring` / `_assignment` / `_partition` / `_ratio`, `most_frequent_bitstring` | Best and most frequent sample. Not the expected value |
| | `best_observed_is_exact_optimum` | Explicit comparison \|best_observed_cut − optimum\| ≤ 1e-9. `None` if no exact reference |
| | `sampled_optimal_frequency` | Fraction of shots whose cut equals the optimum: the **finite-shot success frequency**. It estimates `optimal_solution_probability` but isn't the same quantity. `None` if no exact reference |
| Optimizer status | `optimizer_success` | SciPy termination flag of the best restart. **Does not imply** the Max-Cut optimum was found, and `False` doesn't mean it wasn't. Both cases are tested |
| | `optimizer` | Settings plus per-restart traces (start/end point, nfev, success, message, history) |
| Metadata | `graph`, `bit_order`, `evaluation_mode`, `circuit`, `runtime_seconds` | |

Ratios are `None` when the optimum is 0 or unavailable. **These are logical results and say nothing about either processor.**

### Sources of variation
- **Exhaustive search:** fully deterministic.
- **Statevector quantities:** deterministic for fixed angles in a given numerical environment.
- **Optimizer:** initial points come from `numpy.random.default_rng(seed)`; COBYLA itself is deterministic given its start point and objective.
- **Finite-shot sampling:** `StatevectorSampler(seed=seed)`. The seed is passed through, and a test checks this.
- **Shots objective mode:** in `objective="shots"` mode, every objective evaluation reuses the same sampler seed.
- **`seed=None`:** gives non-reproducible runs.
- **Scope of the reproducibility guarantee:** identical results are tested only for the same seed, config and environment (Python 3.14.7, Qiskit 2.5.2, SciPy 1.18.1, NumPy 2.5.4, Windows). Agreement across library versions, platforms or backends is **not** guaranteed.

## 5. Interface for teammates

```python
from src.problem import load_graph, solve_exact, bitstring_to_assignment, cut_value
from src.qaoa_solver import build_qaoa_circuit, logical_circuit_for_routing, decode_counts, run_qaoa, QAOAConfig

graph = load_graph()                             # final team-selected G6 (6 logical qubits)
qaoa  = build_qaoa_circuit(graph, p=1)          # QAOACircuit: .circuit, .gammas, .betas, .metadata()
bound = qaoa.bind(gammas, betas)                 # or logical_circuit_for_routing(graph, p, gammas, betas)
```

**Member 4 (routing):**
- Build the logical circuit **once** and transpile that same object for Processor A and for Processor B, with identical transpiler settings and seeds.
- The circuit contains only `h`, `rzz`, `rx` and `measure`. It has no topology assumptions and is never transpiled inside `qaoa_solver.py`.
- Keep the measurements in the circuit when transpiling. The explicit `measure(q[i], meas[i])` wiring survives layout and routing, so counts from the compiled circuit still decode in logical vertex order with `decode_counts(graph, counts)`.
- This is tested with a generic *test-only* line coupling map that forces SWAPs. That map is **not** Processor A or B.
- Logical baselines for overhead ratios are in `qaoa.metadata()["logical_depth"]` and `["logical_two_qubit_gates"]`.

**Member 5 (metrics):** use `QAOAResult.to_dict()` for the logical-quality columns, and Member 4's output for the compiled metrics (depth, 2-qubit count, SWAPs, physical qubits).
- Lower SWAP count does not imply better Max-Cut quality. They are separate metrics.
- Noiseless logical quality is the same on A and B by construction. A quality difference between them can only come from a noisy simulation, if the team implements one.

**Member 2 (processors):** this module needs nothing from you. The processor JSONs are consumed only by Member 4.
- Linn's *candidate* definitions are now in the repository: A has 5 qubits and 4 edges, B has 9 qubits and 12 edges. They are unmodified and structurally tested.
- Open issues are listed in [processor_handoff_notes.md](processor_handoff_notes.md), including the 7-vs-9-qubit discrepancy for B and the lack of a check against the official diagram.
- Width constraint only: the logical circuit uses one qubit per vertex, so a graph must have ≤ 5 vertices for A and ≤ 9 for B. **G6 needs 6 qubits, so it does not fit the candidate 5-qubit Processor A as defined.** Report this limitation; don't shrink G6 to fit. No A/B comparison has been run.

## 6. Reproduction

```
.venv\Scripts\python -m pytest -q                       # pytest.ini makes this work from any directory
.venv\Scripts\python -m scripts.run_qaoa_g6 --p 1 --seed 7   # G6 sample run, prints JSON
.venv\Scripts\python -m scripts.run_qaoa_synthetic
```

Dependencies are pinned in `requirements.txt`.

The experiment script writes `results/tables/synthetic_logical_qaoa.json` and `..._summary.csv`.

## 7. Limitations

- **Team-selected benchmark.** G6 is our final benchmark, but not organizer-provided; the kit leaves the choice of graph to teams. The `synthetic_*` graphs are for testing only.
- **Exact methods are small-n.** Exact search and statevector evaluation are exponential. They are practical up to roughly 20 vertices (the guard), and in practice the target is ≤ 9 qubits, matching Processor B.
- **Local optimizer.** COBYLA can stop in a local optimum or at `maxiter`. In the synthetic run at p ≥ 2, nfev reached 200, so those runs did not converge. Raise `maxiter` or `n_restarts` if needed.
- **Heuristic γ range.** The default initial-γ range is a heuristic; the γ period depends on the weights. It is an initialization range only, not an optimizer bound.
- **No G6 QAOA results committed.** Tests exercise G6, but no G6 experiment output has been generated or reported yet.
- **Noiseless only.** No noise model is used. These results make no claim of quantum advantage.
