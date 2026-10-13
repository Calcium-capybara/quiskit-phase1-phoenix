# Processor A/B handoff notes (from Member 2, Linn)

**Status: CANDIDATE definitions. Not yet verified against the official challenge diagrams.**
Integrated by Member 3 on 2026-10-11, without modifying the files.

## Files and provenance

The source is `Linn_Handoff.zip`, SHA-256 `1fc38617595aa50f39d014c18c1876ec61b84a6f00fececbf6055b5ed93d5771`. Its four files were copied into the repository byte-for-byte, replacing empty 0-byte placeholders:

| Repository path | SHA-256 |
|---|---|
| `processors/processor_A.json` | `48007194e172435f0b75b92e462a1a636070a6d3e3fc18e52b5aa1176448de7e` |
| `processors/processor_B.json` | `65c23600b97904e095ff8858ee885045558053eb3fd483e03d7bea069bd48115` |
| `src/processors.py` | `3ab334674b9fdb9cbae96318ebad88dbb9f8035820ad179e1a61397b16679cf3` |
| `docs/handoff/Linn_Handoff.txt` (original note) | `0253cdfd691399a83ae7280ac3b0d1b9b3a1de4653e0a326865ef4f5fd060726` |

**Update after merging `origin/main`.** Teammates had also committed these files to GitHub.
- `src/processors.py` is identical to the zip.
- The repository JSON files now come from `origin/main` and differ from the zip **only by a trailing newline**. The topology is unchanged, but the two JSON checksums above no longer match the committed files.
- The team's `tests/test_processors.py` (by Calcium-capybara) is kept as-is.
- Member 3's structural tests live in `tests/test_processor_configs.py`.

## Candidate topologies

| | Processor A | Processor B |
|---|---|---|
| Qubits | 5 | 9 |
| Unique undirected edges | (0,1) (1,2) (1,3) (3,4) | (0,1) (0,8) (1,2) (1,6) (2,3) (2,5) (2,6) (3,4) (4,5) (5,6) (6,7) (6,8) |
| Directed entries | 8 (each edge in both directions) | 24 (each edge in both directions) |
| Structure | Tree branching at qubit 1 (not a linear chain); no triangles | Connected; all 9 qubits used; triangles (1,2,6) and (2,5,6); qubit 6 has degree 5 |

`tests/test_processor_configs.py` checks the following:
- **Qubit counts and indices:** 5 and 9 qubits, all indices valid, no self-loops.
- **Edges:** no duplicate directed entries, every edge has its reverse, and the unique edges equal the lists above.
- **Coverage:** every qubit is used and each graph is connected.
- **Qiskit:** `CouplingMap` accepts both maps (a structural check only).
- **Loader errors:** the loader rejects malformed configurations.
- **Module boundary:** `problem.py` and `qaoa_solver.py` do not import processor or routing code.

## Open issues (not resolved here, to avoid guessing)

1. **Processor B qubit count: 9 vs 7.** An older guideline describes Processor B as 7 qubits. Linn's definition, and the organizers' later clarification that B "must use all nine physical qubits", say 9. This needs confirming against the official Challenge Kit.
2. **Not verified against the official diagram.** `Linn_Handoff.txt` calls the topology "verified", but that refers to internal consistency checks: index range, no duplicates, reverse entries. No check against the official Processor A/B diagrams is documented. The organizers have said Processor A is *not* a linear chain; the candidate A (a tree branching at qubit 1) is consistent with that, but that alone does not confirm it.
3. **Unspecified:** basis/native gates, noise and calibration parameters, port assignments, and the final JSON schema. The handoff says these are "unspecified because authoritative values have not been established". No values have been invented.
4. **Qiskit integration not done.** The handoff says it has not been tested beyond the structural `CouplingMap` check above. Conversion to a Qiskit `Target`/`CouplingMap` for transpilation belongs to the routing pipeline (Member 4).
5. **Ownership of `src/processors.py`.** The earlier team plan assigned it to Member 4; Linn (Member 2) delivered it. By team decision, it is kept as Linn's file, unchanged.

## Boundary with Member 3's modules

`src/problem.py` and `src/qaoa_solver.py` use no processor data. The logical QAOA circuit has one qubit per graph vertex, so an n-vertex graph needs n ≤ 5 to fit Processor A and n ≤ 9 to fit Processor B. **No Processor A/B comparison has been run or interpreted yet.**
