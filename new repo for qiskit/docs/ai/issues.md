# Technical Issues & Resolution Log
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix  
**Maintainers:** Member 1 (Lead), Member 3 (QAOA), Member 4 (Routing)  
**Scope:** Log of genuine technical bugs, discrepancies, compiler quirks, and human-guided resolutions.

---

### Issue 01: Three-Way Processor A Coupling Map Conflict
- **Severity:** High (Core Architecture)
- **Component:** `processors/processor_A.json`, `processor_A.py`, `docs/processor_handoff_notes.md`
- **Symptom:** The generator script `processor_A.py` emitted a linear chain `[(0,1),(1,2),(2,3),(3,4)]`, while `processor_A.json` held a hub at 0 `[(0,1),(0,2),(0,3),(3,4)]`, and handoff notes described branching at 1 `[(0,1),(1,2),(1,3),(3,4)]`.
- **Root Cause:** Multiple team members explored different candidate topologies without synchronizing back to the JSON generator.
- **Team Resolution:** Cross-referenced Page 1 and Page 2 of the official Phase 1 Master Guide and its visual stage diagram (`Processor models across stages`). The visual diagram depicts a central top hub with 3 downward branches and a stem. The team formally ratified the T-shaped branching tree `[(0,1), (0,2), (0,3), (3,4)]`. Updated JSON, tests, and documentation.

---

### Issue 02: Latent Big-Endian Bit Indexing in Shots-Mode QAOA Expectation
- **Severity:** Critical (Algorithmic Correctness)
- **Component:** `src/qaoa_solver.py` (Line 478)
- **Symptom:** In `objective="shots"` mode, the expectation value calculation evaluated `int(k, 2)` directly on Qiskit measurement keys.
- **Root Cause:** Qiskit measurement keys are little-endian strings where the rightmost character is qubit 0. Evaluating `int(k, 2)` interprets the string as big-endian (where the leftmost character is the most significant bit). For non-symmetric graph cuts, this evaluated incorrect cut indices.
- **Team Resolution:** The team audited the evaluation function and replaced the raw integer cast with:
  ```python
  cut_values[assignment_to_index(bitstring_to_assignment(k, qaoa.graph.num_nodes, order="qiskit"))]
  ```
  Verified that non-symmetric bitstring tests (`test_both_bit_orders_on_g6_non_palindromic_key`) catch any reversal.

---

### Issue 03: Processor A Physical Capacity Wall on 6-Node Benchmarks
- **Severity:** Architectural Limitation
- **Component:** `src/routing.py`
- **Symptom:** Transpiler threw internal errors or attempted mapping when compiling 6-qubit QAOA onto 5-qubit Processor A.
- **Root Cause:** Physical qubit budget of Processor A is strictly 5 qubits, but PS-O1 specifies a 6-node weighted graph.
- **Team Resolution:** Rather than altering the benchmark to fit hardware or padding results, the team implemented a strict pre-compilation `CapacityError`. In all comparative reports, Processor A is explicitly marked `CAPACITY_EXCEEDED`, demonstrating honest algorithmic analysis.

---

### Issue 04: Transpiler Seed Sensitivity in Routing Depth
- **Severity:** Moderate (Reproducibility)
- **Component:** `src/routing.py`
- **Symptom:** Running `transpile` with different random seeds produced variance in CX count (between 21 and 27 CX) on heavy-hex Processor B.
- **Root Cause:** SABRE layout and routing heuristics depend on stochastic random initializations.
- **Team Resolution:** Pinned `seed_transpiler=42` as the canonical benchmark baseline per Page 2 of the Master Guide, ensuring identical bit-for-bit compiled depths across all runs.
