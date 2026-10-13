# Architecture Decision Records (ADRs)
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix (Six Members)  
**Track 2:** Quantum Optimization (QAOA / QUBO)  
**Standard:** Lightweight ADR Format (Context, Decision, Consequences)

---

### ADR 001: Separation of Logical QAOA from Physical Processor Topologies
- **Status:** Accepted (2026-10-11)
- **Context:** The team needed to structure variational algorithm formulation vs hardware transpilation so Member 3 (QAOA/Algorithm) and Member 4 (Routing/Architecture) could work concurrently without circular dependencies.
- **Decision:** Keep `src/problem.py` and `src/qaoa_solver.py` strictly classical and topology-agnostic. The logical circuit outputs standard parameterised circuits with virtual qubits ($0 \dots n-1$). Transpilation onto hardware coupling maps is isolated exclusively to `src/routing.py`.
- **Consequences:** Clean separation of concerns. Algorithm tests can run independent of hardware graphs, and transpilation settings can be swept across multiple candidate architectures (A, B, and D) without altering ansatz code.

---

### ADR 002: Mathematical Sign & Gate Convention for QAOA Cost Hamiltonian
- **Status:** Accepted (2026-10-12)
- **Context:** Standard literature presents varying definitions of $R_{ZZ}$ gate angles, often omitting factors of 2 or negative signs relative to Qiskit's internal convention $R_{ZZ}(\theta) = \exp(-i \frac{\theta}{2} Z \otimes Z)$.
- **Decision:** Explicitly derive gate mappings:
  $$\hat{C} = \sum_{(u,v)} w_{uv} \frac{I - Z_u Z_v}{2} \implies U_C(\gamma) = e^{-i \gamma \hat{C}} \equiv \text{phase} \times R_{ZZ}(-w_{uv} \gamma)$$
  $$U_B(\beta) = e^{-i \beta \sum X} \equiv R_X(2 \beta)$$
- **Consequences:** All cost rotations are realized as `rzz(-w * gamma, u, v)` and mixer rotations as `rx(2 * beta, q)`. Verified via `scipy.linalg.expm` and closed-form single-edge tests in `tests/test_qaoa_solver.py`.

---

### ADR 003: Little-Endian Bitstring Decoding and Explicit Bit-Order Interface
- **Status:** Accepted (2026-10-12)
- **Context:** Qiskit measurement keys report classical register bits in little-endian order (rightmost bit = bit 0), whereas graph algorithms and human reading order place vertex 0 on the left. Symmetrical benchmark bitstrings (like `010101`) can mask reversal errors.
- **Decision:** Mandate an explicit `order="qiskit"` or `order="vertex"` keyword argument in all bitstring conversion functions (`bitstring_to_assignment`, `assignment_to_bitstring`). Hardened tests using non-symmetric bitstrings (`000011` vs `110000`).
- **Consequences:** Prevents silent bit-order reversal bugs across measurement decoding, cut evaluation, and probability extraction.

---

### ADR 004: Adopting the Official Erdös-Rényi Benchmark alongside Team G6
- **Status:** Accepted (2026-10-13)
- **Context:** The official PS-O1 task description specifies `nx.erdos_renyi_graph(n=6, p=0.5, seed=42)` with integer edge weights 1–5, while the team had also frozen a candidate benchmark G6.
- **Decision:** Fully support and benchmark both graphs:
  1. `data/raw/erdos_renyi_6.json`: Exact official programmatic benchmark instance (optimal cut = 16.0).
  2. `data/raw/weighted_maxcut_g6.json`: Team-selected fixed 6-vertex benchmark (optimal cut = 24.0).
- **Consequences:** 100% compliance with official problem statement requirements while preserving comprehensive historical team benchmarks.

---

### ADR 005: Creative Architecture Exploration & Processor D Co-Design
- **Status:** Accepted (2026-10-13)
- **Context:** Organizers emphasized: *"There is no single 'correct' coupling map. Your architecture is part of your solution."* Analysis of heavy-hex Processor B revealed a SWAP bottleneck due to limited planar chords when routing dense 60% interaction graphs.
- **Decision:** In addition to the reference baseline (Processor A: 5q T-branching tree) and contemporary reference (Processor B: 9q heavy-hex), design **Processor D** (9 physical qubits, 15 edges, ring-with-chord triangular core).
- **Consequences:** Reduced hardware diameter from 3 to 2, lowered the Exploratory Connectivity Mismatch Score ($M$) by $69.2\%$, and demonstrated meaningful algorithm-architecture co-design within Phase 1 limits ($\le 12$ qubits).

---

### ADR 006: Pre-Compilation Capacity Validation and Honest Failure Reporting
- **Status:** Accepted (2026-10-13)
- **Context:** Processor A provides 5 physical qubits, while the challenge requires a 6-node benchmark graph.
- **Decision:** Explicitly catch and report the physical capacity barrier using `CapacityError` in `src/routing.py`, recording `CAPACITY_EXCEEDED` in comparative result tables.
- **Consequences:** Maintains strict scientific integrity. No artificial truncation, fake 5-qubit graph scaling, or fabricated execution results.
