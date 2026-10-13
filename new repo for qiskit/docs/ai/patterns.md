# Engineering Design Patterns & Best Practices
## Qiskit Fall Fest 2026 — Phase 1: PS-O1 (Weighted Max-Cut QAOA)

**Team:** Phoenix  
**Scope:** Architectural and software engineering patterns established by the team for quantum algorithm development.

---

### Pattern 1: Classical-Quantum Boundary Decoupling
- **Pattern:** Strict isolation of classical graph manipulation (`src/problem.py`) from quantum circuit construction (`src/qaoa_solver.py`) and hardware transpilation (`src/routing.py`).
- **Rationale:** `src/problem.py` has zero Qiskit dependencies. It can run in pure Python / NumPy, enabling lightning-fast classical exhaustive verification and testing without importing heavy quantum frameworks.

### Pattern 2: Explicit-Ordering Safety Envelopes
- **Pattern:** Mandatory keyword-only arguments for bitstring transformations:
  ```python
  bitstring_to_assignment(bitstring, num_nodes, *, order: "qiskit" | "vertex")
  ```
- **Rationale:** Prevents silent bit-reversal bugs caused by defaulting or relying on ambiguous little-endian vs big-endian representations. Any call omitting `order` fails at runtime.

### Pattern 3: Pre-Flight Capacity Validation
- **Pattern:** Fail-fast capacity checks executed prior to launching compiler passes:
  ```python
  if circuit.num_qubits > processor["num_qubits"]:
      raise CapacityError(...)
  ```
- **Rationale:** Avoids expensive or obscure transpiler graph embedding failures and clearly articulates physical architecture limits directly to the user.

### Pattern 4: Dual Metric Reporting (Theoretical vs Empirical)
- **Pattern:** Clear distinction between:
  1. Exact statevector metrics (infinite shots, exact probabilities).
  2. Sampled shot counts (finite shots, stochastic frequencies).
  3. Exploratory topological metrics (distance heuristics like $M$).
  4. Real compiled metrics (actual CX, depth, SWAP counts).
- **Rationale:** Eliminates ambiguity in benchmarking. Judges can immediately discern theoretical statevector support from real measured counts.
