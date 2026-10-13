# AI Collaboration & Prompt Log (`ai_prompt.md`)
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix  
**Documentation Purpose:** Transparency log documenting AI pair-programming interactions, prompt engineering strategies, verification steps, and human-in-the-loop validation during Phase 1.

---

## 1. AI Collaboration Strategy & Philosophy

Per the official organizer guidance:
> *"Also those who are taking leverage of AI, i would recommend to also add `ai_prompt.md`, `adrs.md`, etc under `docs/` which contains all the ai prompts, decisions, etc which you have used throughout the phase which may give you a higher edge compared to other participants."*

Our team utilized AI coding assistants as **pair programmers, code auditors, and test generators**. All scientific claims, mathematical proofs, architecture designs, and benchmark results were independently verified through automated test suites and classical ground-truth scripts.

---

## 2. Structured Prompt Log & Verification Trajectory

### Prompt Interaction 1: Repository Audit & Consistency Verification
- **User Prompt Objective:** Perform a read-only audit of the existing codebase to identify discrepancies, unverified claims, and topology conflicts.
- **AI Task Executed:** 
  - Scanned repository files recursively.
  - Checked `processor_A.py` vs `processor_A.json` vs handoff notes.
  - Formally reported 3 distinct competing topologies for Processor A and missing test files.
- **Human Verification:** Verified that Processor A's generator script had generated a linear chain while the JSON described a branching tree. Consulted the official Master Guide to resolve the conflict.

### Prompt Interaction 2: Mathematical Derivation & QAOA Sign Convention
- **User Prompt Objective:** Ensure QAOA Hamiltonian decomposition correctly aligns with Qiskit 2.x gate definitions without global-phase or sign errors.
- **AI Task Executed:**
  - Formulated:
    $$U_C(\gamma) = \exp\left(-i \gamma \sum_{(u,v)} w_{uv} \frac{I - Z_u Z_v}{2}\right) \implies R_{ZZ}(-w_{uv} \gamma)$$
  - Generated unit test `test_cost_unitary_equals_exp_minus_i_gamma_C` comparing the circuit unitary against `np.diag(np.exp(-1j * gamma * cut_values))`.
- **Human Verification:** Confirmed that flipping the sign caused 4 unit tests to fail immediately, proving the test suite was non-vacuous.

### Prompt Interaction 3: Member 4 Routing Engine & Capacity Enforcement
- **User Prompt Objective:** Take over Member 4's uncompleted routing module and build processor-aware compilation with strict capacity pre-checking.
- **AI Task Executed:**
  - Implemented `validate_capacity(circuit, processor_config)` raising `CapacityError`.
  - Implemented `compile_for_processor()` collecting logical vs compiled depth, CX counts, and layout metadata.
  - Implemented `compare_processors()` generating comparative tables.
- **Human Verification:** Executed tests ensuring 6-qubit circuits onto 5-qubit Processor A cleanly report capacity failure rather than crashing or faking execution.

### Prompt Interaction 4: Topological Connectivity Mismatch Score ($M$)
- **User Prompt Objective:** Implement and evaluate an exploratory metric assessing physical graph distance vs logical edge interaction density (Prompt 5B).
- **AI Task Executed:**
  - Implemented:
    $$M = \frac{\sum_{(i,j)\in E} w_{ij} \left[ d(\pi(i), \pi(j)) - 1 \right]}{\sum_{(i,j)\in E} w_{ij}}$$
  - Implemented edge-case handling for incomplete mappings and disconnected components.
  - Wrote 6 unit tests in `tests/test_metrics.py`.
- **Human Verification:** Manually computed distances on small linear chains to verify calculation accuracy before applying to 9-qubit graphs.

### Prompt Interaction 5: Architectural Creativity & Processor D Co-Design
- **User Prompt Objective:** Address the organizer challenge: *"Your architecture is part of your solution"* by creating Processor D to alleviate heavy-hex routing bottlenecks.
- **AI Task Executed:**
  - Designed 9-qubit ring-with-chord triangular-core coupling map (`processors/processor_D.json`).
  - Added chord edges between outer ring and inner core to reduce diameter from 3 to 2.
  - Benchmarked empirical routing overhead and verified $69.2\%$ reduction in $M$ score.
- **Human Verification:** Verified that Processor D contains exactly 9 qubits (under the 12-qubit Phase 1 limit) and forms a valid bi-directional coupling map.

---

## 3. Human-in-the-Loop Quality Control Summary

| Component | AI Role | Human Validation & Decision |
|---|---|---|
| **Topology Resolution** | Identified file discrepancies across branches | Inspected official Master Guide screenshot; selected T-branching tree matching Page 1/2 visual diagram. |
| **Objective Validation** | Implemented vectorised NumPy cut calculator | Independently hand-calculated cut values for non-symmetric assignments. |
| **Bug Detection** | Audited `src/qaoa_solver.py` | Caught and repaired latent indexing bug in `objective="shots"` mode where bitstring keys were incorrectly indexed. |
| **Capacity Handling** | Proposed options (error vs padding) | Mandated strict scientific integrity: raise `CapacityError` and report `CAPACITY_EXCEEDED`. |
| **Test Coverage** | Generated parametrized pytest cases | Executed full test suite (333 tests) in clean virtual environment. |
