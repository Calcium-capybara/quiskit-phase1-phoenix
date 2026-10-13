# Project Plan Evolution & Iteration History
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix  
**Scope:** Chronological evolution of team project roadmap, architectural pivots, and scope management.

---

### Phase 1 Initial Blueprint (Sprint 1: Days 1–3)
- **Focus:** Problem understanding, mathematical formulation, and initial QAOA exploration on 4-qubit toy graphs.
- **Milestones Completed:**
  - Implemented initial Max-Cut objective.
  - Explored $p=1$ logical circuits using basic Qiskit primitives.
  - Defined candidate 5-qubit and 9-qubit processor stubs.
- **Bottlenecks Identified:** Lack of standardized benchmark graph; Member 4 routing stalled; conflicting processor topologies across different branches.

---

### Architecture Refinement & Master Guide Synchronization (Sprint 2: Days 4–6)
- **Trigger:** Release of the SRM University-AP **QFF 2026 Phase 1 Master Guide**.
- **Pivots & Team Decisions:**
  1. **Phase Scope Clarification:** Confirmed that Processor C and D are strictly reserved for Phase 2 finalists; Phase 1 focus is strictly Processor A, Processor B, and creative custom co-design.
  2. **Benchmark Unification:** Added the official programmatic NetworkX generator `nx.erdos_renyi_graph(n=6, p=0.5, seed=42)` alongside the frozen team G6 benchmark.
  3. **Capacity Realization:** Formally established that 6-qubit benchmarks cannot execute on 5-qubit Processor A; codified `CapacityError` into team requirements.
  4. **Creative Architecture Exploration:** Designed **Processor D** (9-qubit ring-with-chord) to explore how adding cross-chords relieves planar routing bottlenecks on dense graphs.

---

### Clean Reconstruction & Submission Hardening (Sprint 3: Final Sprint)
- **Focus:** Complete repository rebuild, strict compliance with the organizer folder layout, test suite hardening, and independent verification.
- **Milestones Completed:**
  - Standardized on authoritative layout: `data/`, `processors/`, `src/`, `results/`, `assets/`, `docs/`.
  - Reconstructed Member 4 routing engine and Member 5 metrics modules.
  - Implemented Exploratory Connectivity Mismatch Score ($M$).
  - Built comprehensive 333-test automated test suite passing at 100%.
  - Fully verified all classical optima and numerical outputs.
