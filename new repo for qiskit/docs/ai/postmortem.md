# Technical Postmortem & Engineering Retrospective
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix  
**Scope:** Engineering postmortem analyzing team challenges, design mistakes, lessons learned, and guidelines for Phase 2.

---

### What Went Wrong During Early Iterations
1. **Branch Divergence on Hardware Topologies:**
   - Early team members independently modified `processor_A.py` and `processor_A.json` without automated schema or regression checks. This created three competing topologies in git history (chain, hub-at-0, hub-at-1).
   - *Lesson:* Every hardware configuration file must have an automated test asserting edge list, node count, and bi-directionality, run in continuous integration.

2. **Silent Failure in Reversed Bitstrings:**
   - The team's initial benchmark G6 had complementary optimal bitstrings (`010101` and `101010`) that happen to be palindromic under reversal. This masked a latent indexing bug in early code because reversed strings evaluated to the exact same cut.
   - *Lesson:* Benchmark validation must always test non-symmetric edge cases (e.g., `000011` vs `110000`) to guarantee bit-endianness correctness.

3. **Incomplete Role Ownership (Member 4 Stalling):**
   - Transpilation and routing were left as empty stubs until late in the timeline, delaying empirical SWAP counts and depth measurements.
   - *Lesson:* Integration interfaces (like `logical_circuit_for_routing`) should be mock-tested early so downstream metrics and tables do not stall.

---

### What Went Right
1. **Defensive Mathematical Implementation:**
   - Writing tests that directly compared circuit unitaries against `scipy.linalg.expm` caught subtle minus sign and angle errors early.
2. **Honest Capacity Handling:**
   - Rather than hacking the 6-qubit graph to fit on 5-qubit Processor A, the team documented the capacity barrier as a primary scientific finding, which strengthens the architectural narrative.
3. **Creative Custom Architecture:**
   - Designing Processor D directly in response to measured SWAP bottlenecks on Processor B provided clear evidence of algorithm-architecture co-design.

---

### Recommendations for Phase 2 (Offline Finalists)
1. Incorporate real device calibration data and native pulse-level gate durations into depth calculations.
2. Explore error mitigation techniques (Zero-Noise Extrapolation, Readout Mitigation) on noisy simulators.
3. Scale custom topologies to 13–16 qubits (Processor C and beyond).
