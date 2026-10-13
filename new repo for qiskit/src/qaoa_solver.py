"""Logical QAOA for weighted Max-Cut (Member 3).

Separation of concerns
----------------------
* Graph/objective ................ :mod:`src.problem` (pure classical).
* Logical circuit construction ... :func:`build_qaoa_circuit` (this module).
* Parameter optimisation ......... :func:`optimize_parameters` / :func:`run_qaoa`.
* Sampling and decoding .......... :func:`sample_counts`, :func:`decode_counts`.
* Transpilation/routing .......... NOT here -- Member 4 (``src/routing.py``)
  compiles the circuit returned by :func:`build_qaoa_circuit` for each processor.
  This module never imports processor definitions or coupling maps.

Mathematics and sign convention
-------------------------------
Cost Hamiltonian (diagonal in the computational basis)::

    C_hat = sum_{(u,v) in E} w_uv (I - Z_u Z_v) / 2

For a basis state |x>, Z_u Z_v |x> = (+1 if x_u == x_v else -1) |x>, so
(I - Z_u Z_v)/2 |x> = [x_u != x_v] |x> and  <x|C_hat|x> = C(x), the weighted cut.
Hence <psi|C_hat|psi> = sum_x |<x|psi>|^2 C(x) is the expected cut value.

QAOA state (depth p)::

    |gamma, beta> = prod_{l=1..p} U_B(beta_l) U_C(gamma_l) |+>^n
    U_C(gamma) = exp(-i gamma C_hat)
    U_B(beta)  = exp(-i beta sum_j X_j)

Gate realisation (verified against Qiskit's definitions in the tests):

* Qiskit: RZZ(theta) = exp(-i theta/2 Z Z), RX(theta) = exp(-i theta/2 X).
* exp(-i gamma w (I - Z Z)/2) = exp(-i gamma w/2) * exp(+i gamma w/2 Z Z)
  = (global phase) * RZZ(-w gamma).
  So each edge contributes ``rzz(-w * gamma)`` and the global phase is dropped.
* exp(-i beta X) = RX(2 beta), applied to every qubit.

Many tutorials instead apply RZZ(2 w gamma') (i.e. exp(-i gamma' w ZZ), minimising
sum w ZZ). That is the same family of states with gamma = -2 gamma'; numbers
from such sources must be converted before reuse here.
Zero-weight edges contribute the identity and are skipped (no gate emitted).

Qubit / bit ordering (single convention used everywhere)
--------------------------------------------------------
vertex i  ->  logical qubit q[i]  ->  classical bit meas[i]  ->  character
``bitstring[n-1-i]`` of the Qiskit counts key (rightmost character = vertex 0).
Decode with ``problem.bitstring_to_assignment(key, n, order="qiskit")``.
Measurements are explicit ``measure(q[i], meas[i])`` so that after Member 4's
transpilation (which remaps qubits but preserves the qubit->clbit wiring of the
measure instructions) counts keys remain in *logical vertex* order.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from src.problem import (
    CUT_ATOL,
    DEFAULT_MAX_EXACT_NODES,
    ExactSolution,
    WeightedGraph,
    all_cut_values,
    approximation_ratio,
    assignment_to_index,
    bitstring_to_assignment,
    cut_value,
    partition,
    solve_exact,
)

QREG_NAME = "q"
CREG_NAME = "meas"
BIT_ORDER = "qiskit"


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------


def cost_hamiltonian(graph: WeightedGraph) -> SparsePauliOp:
    """``C_hat = sum_{(u,v)} w_uv (I - Z_u Z_v)/2`` as a ``SparsePauliOp`` on ``n`` qubits.

    Qubit ``i`` is vertex ``i``. Its diagonal equals :func:`problem.all_cut_values`.
    """
    n = graph.num_nodes
    terms: list[tuple[str, list[int], float]] = [("", [], 0.5 * graph.total_weight)]
    terms += [("ZZ", [u, v], -0.5 * w) for u, v, w in graph.edges]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


# ---------------------------------------------------------------------------
# Logical circuit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAOACircuit:
    """A parameterised, topology-independent logical QAOA circuit.

    Attributes:
        circuit: The parameterised circuit (with measurements if built so).
        gammas: Cost-layer parameters, ``gammas[l]`` for layer ``l``.
        betas: Mixer-layer parameters, ``betas[l]`` for layer ``l``.
        p: QAOA depth.
        graph: The problem graph the circuit was built from.
        measured: Whether measurement instructions are included.
        vertex_to_qubit: ``vertex_to_qubit[i]`` is the logical qubit index of vertex ``i``.
        qubit_to_clbit: ``qubit_to_clbit[q]`` is the classical bit measuring qubit ``q``.
        bit_order: Bitstring order of counts keys (``"qiskit"``: rightmost = vertex 0).
    """

    circuit: QuantumCircuit
    gammas: ParameterVector
    betas: ParameterVector
    p: int
    graph: WeightedGraph
    measured: bool
    vertex_to_qubit: tuple[int, ...]
    qubit_to_clbit: tuple[int, ...]
    bit_order: str = BIT_ORDER

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits

    def parameter_dict(self, gammas: Sequence[float], betas: Sequence[float]) -> dict:
        """Map concrete angle values onto this circuit's ``Parameter`` objects."""
        gammas, betas = _check_angles(gammas, betas, self.p)
        mapping = {self.gammas[l]: gammas[l] for l in range(self.p)}
        mapping.update({self.betas[l]: betas[l] for l in range(self.p)})
        return mapping

    def bind(self, gammas: Sequence[float], betas: Sequence[float]) -> QuantumCircuit:
        """Return a copy of the circuit with all angles bound to numbers."""
        # strict=False: a gamma has no gate if the graph has no (non-zero) edges.
        return self.circuit.assign_parameters(self.parameter_dict(gammas, betas), strict=False)

    def metadata(self) -> dict[str, Any]:
        """Interface information for routing/metrics owners (JSON-serialisable)."""
        ops = self.circuit.count_ops()
        return {
            "p": self.p,
            "num_qubits": self.num_qubits,
            "num_clbits": self.circuit.num_clbits,
            "measured": self.measured,
            "vertex_to_qubit": list(self.vertex_to_qubit),
            "qubit_to_clbit": list(self.qubit_to_clbit),
            "bit_order": self.bit_order,
            "parameter_names": [prm.name for prm in [*self.gammas, *self.betas]],
            "logical_depth": self.circuit.depth(),
            "logical_two_qubit_gates": int(ops.get("rzz", 0)),
            "logical_op_counts": {k: int(v) for k, v in ops.items()},
            "graph": self.graph.to_dict(),
        }


def build_qaoa_circuit(
    graph: WeightedGraph, p: int = 1, *, measure: bool = True, barriers: bool = False
) -> QAOACircuit:
    """Build the logical depth-``p`` QAOA circuit for ``graph``.

    The circuit uses only ``h``, ``rzz``, ``rx`` (and ``measure``); it is NOT
    transpiled and makes no assumption about connectivity.

    Args:
        graph: Weighted Max-Cut instance.
        p: Number of QAOA layers (>= 1).
        measure: Append ``measure(q[i], meas[i])`` for every vertex ``i``.
        barriers: Insert barriers between layers (visual aid; barriers also
            constrain transpiler optimisation, so leave False for routing studies).
    """
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError(f"p must be an int >= 1, got {p!r}")
    p = int(p)
    n = graph.num_nodes
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)

    qreg = QuantumRegister(n, QREG_NAME)
    if measure:
        creg = ClassicalRegister(n, CREG_NAME)
        qc = QuantumCircuit(qreg, creg, name=f"qaoa_p{p}")
    else:
        qc = QuantumCircuit(qreg, name=f"qaoa_p{p}")

    qc.h(qreg)
    for layer in range(p):
        if barriers:
            qc.barrier()
        append_cost_layer(qc, graph, gammas[layer])
        if barriers:
            qc.barrier()
        append_mixer_layer(qc, betas[layer])
    if measure:
        if barriers:
            qc.barrier()
        for i in range(n):
            qc.measure(qreg[i], creg[i])

    qc.metadata = {"problem": "weighted_maxcut", "graph_name": graph.name, "graph_source": graph.source,
                   "p": p, "bit_order": BIT_ORDER}
    return QAOACircuit(
        circuit=qc,
        gammas=gammas,
        betas=betas,
        p=p,
        graph=graph,
        measured=measure,
        vertex_to_qubit=tuple(range(n)),
        qubit_to_clbit=tuple(range(n)) if measure else (),
    )


def append_cost_layer(qc: QuantumCircuit, graph: WeightedGraph, gamma: Any) -> None:
    """Append ``U_C(gamma) = exp(-i gamma C_hat)`` (up to global phase): ``rzz(-w*gamma)`` per edge."""
    for u, v, w in graph.edges:
        if w != 0.0:
            qc.rzz(-w * gamma, u, v)


def append_mixer_layer(qc: QuantumCircuit, beta: Any) -> None:
    """Append ``U_B(beta) = exp(-i beta sum_j X_j)``: ``rx(2*beta)`` on every qubit."""
    for q in range(qc.num_qubits):
        qc.rx(2 * beta, q)


def _check_angles(gammas: Sequence[float], betas: Sequence[float], p: int) -> tuple[list[float], list[float]]:
    g = [float(x) for x in np.atleast_1d(gammas)]
    b = [float(x) for x in np.atleast_1d(betas)]
    if len(g) != p or len(b) != p:
        raise ValueError(f"expected {p} gammas and {p} betas, got {len(g)} and {len(b)}")
    if not all(math.isfinite(x) for x in g + b):
        raise ValueError("angles must be finite")
    return g, b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def statevector_probabilities(qaoa: QAOACircuit, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """Exact basis-state probabilities, indexed by ``k = sum_i x_i 2**i``."""
    circ = qaoa.bind(gammas, betas)
    if qaoa.measured:
        circ = circ.remove_final_measurements(inplace=False)
    return Statevector(circ).probabilities()


def statevector_expectation(
    qaoa: QAOACircuit, gammas: Sequence[float], betas: Sequence[float], *, cut_values: np.ndarray | None = None
) -> float:
    """Exact ``<gamma,beta| C_hat |gamma,beta>`` = expected cut value."""
    if cut_values is None:
        cut_values = all_cut_values(qaoa.graph)
    return float(statevector_probabilities(qaoa, gammas, betas) @ cut_values)


def sample_counts(
    qaoa: QAOACircuit, gammas: Sequence[float], betas: Sequence[float], *, shots: int = 1024, seed: int | None = None
) -> dict[str, int]:
    """Sample the measured logical circuit with Qiskit's ``StatevectorSampler`` (noiseless).

    Keys are Qiskit-ordered bitstrings (rightmost character = vertex 0).
    """
    if not qaoa.measured:
        raise ValueError("circuit was built with measure=False")
    if shots < 1:
        raise ValueError("shots must be >= 1")
    sampler = StatevectorSampler(seed=seed)
    job = sampler.run([qaoa.bind(gammas, betas)], shots=int(shots))
    return dict(getattr(job.result()[0].data, CREG_NAME).get_counts())


@dataclass(frozen=True)
class DecodedSample:
    bitstring: str
    assignment: tuple[int, ...]
    partition: tuple[tuple[int, ...], tuple[int, ...]]
    cut_value: float
    count: int


def decode_counts(graph: WeightedGraph, counts: Mapping[str, int]) -> list[DecodedSample]:
    """Decode Qiskit-ordered counts into assignments and cut values.

    Sorted by cut value (descending), then count (descending), then bitstring.
    """
    out = []
    for key, c in counts.items():
        x = bitstring_to_assignment(key, graph.num_nodes, order=BIT_ORDER)
        out.append(DecodedSample(key, x, partition(x), cut_value(graph, x), int(c)))
    out.sort(key=lambda s: (-s.cut_value, -s.count, s.bitstring))
    return out


def cut_distribution(samples: Sequence[DecodedSample]) -> dict[float, int]:
    """Histogram ``{cut value: number of shots}`` (ascending cut value)."""
    dist: dict[float, int] = {}
    for s in samples:
        dist[s.cut_value] = dist.get(s.cut_value, 0) + s.count
    return dict(sorted(dist.items()))


def best_cut_with_statevector_probability(
    probabilities: np.ndarray, cut_values: np.ndarray, threshold: float
) -> float | None:
    """Largest cut value among basis states whose statevector probability is ``> threshold``.

    Statevector-derived (exact probabilities, no sampling): it says which cut
    values the QAOA state supports, not what a finite number of shots observed.
    Returns ``None`` if no state exceeds the threshold.

    NOT evidence that QAOA "found" the optimum: any probability above the
    threshold on an optimal assignment is enough, so this usually equals the
    exact optimum even for unconverged or random angles (e.g. the uniform state
    at gamma = beta = 0). Use ``optimal_solution_probability`` to judge how much
    weight the state puts on optimal cuts, and ``best_observed_cut`` for what
    finite sampling actually produced.

    ``threshold`` must be finite and in [0, 1); probabilities must be finite
    and non-negative.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    cut_values = np.asarray(cut_values, dtype=float)
    if probabilities.shape != cut_values.shape:
        raise ValueError("probabilities and cut_values must have the same shape")
    threshold = _require_real("threshold", threshold)
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    mask = probabilities > threshold
    return float(cut_values[mask].max()) if mask.any() else None


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


@dataclass
class QAOAConfig:
    """Experiment settings. These are team choices, NOT official challenge requirements.

    Attributes:
        p: QAOA depth.
        optimizer: Any gradient-free ``scipy.optimize.minimize`` method, e.g.
            ``"COBYLA"`` (default), ``"Nelder-Mead"``, ``"Powell"``.
        maxiter: Iteration limit per restart.
        tol: Optimiser tolerance (passed as ``tol``).
        n_restarts: Number of starting points; the best final value is kept.
        initial_point: Optional ``[gamma_1..gamma_p, beta_1..beta_p]`` for the
            first restart; later restarts are random.
        objective: ``"statevector"`` (exact expectation, default) or ``"shots"``
            (sampled mean cut with ``shots`` per evaluation).
        shots: Shots for final sampling (and for the objective in ``"shots"`` mode).
            Default 4096 = the PS-O1 Challenge Kit setting for reported sampled results.
        seed: Seed for initial points and the sampler.
        gamma_range / beta_range: INITIALISATION ranges only. Random starting
            angles are drawn uniformly from them; they are NOT optimiser bounds.
            ``scipy.optimize.minimize`` is called without ``bounds`` or
            constraints, so COBYLA may move angles outside these ranges, and the
            returned ``gammas``/``betas`` can lie outside them (e.g. a negative
            beta). Range rationale: C_hat is invariant under flipping all bits,
            so the expectation is pi/2-periodic in each beta and [0, pi/2) covers
            every distinct mixer angle; the gamma period depends on the weights
            (2*pi for integer weights), so [0, pi) is a heuristic choice, not a
            mathematically required QAOA range.
        statevector_probability_threshold: Probability above which a basis state
            counts as "supported" by the optimised statevector for
            ``QAOAResult.best_cut_with_statevector_probability`` (strict ``>``).
            The default 1e-12 is far above double-precision round-off in
            probabilities (~1e-16) yet below any probability that matters at
            practical shot counts.
    """

    p: int = 1
    optimizer: str = "COBYLA"
    maxiter: int = 200
    tol: float | None = 1e-6
    n_restarts: int = 3
    initial_point: Sequence[float] | None = None
    objective: str = "statevector"
    shots: int = 4096
    seed: int | None = 1234
    gamma_range: tuple[float, float] = (0.0, math.pi)
    beta_range: tuple[float, float] = (0.0, math.pi / 2)
    statevector_probability_threshold: float = 1e-12

    def validate(self) -> None:
        """Reject invalid settings up front (``TypeError`` for types, ``ValueError`` for values).

        Nothing is coerced or silently repaired: ``bool`` is not accepted where an
        int is required, and non-finite numbers are rejected.
        """
        for name in ("p", "maxiter", "n_restarts", "shots"):
            _require_int(name, getattr(self, name), minimum=1)
        if not isinstance(self.optimizer, str) or not self.optimizer:
            raise TypeError(f"optimizer must be a non-empty str, got {self.optimizer!r}")
        if self.objective not in ("statevector", "shots"):
            raise ValueError(f"objective must be 'statevector' or 'shots', got {self.objective!r}")
        if self.tol is not None:
            _require_real("tol", self.tol)
            if self.tol <= 0:
                raise ValueError(f"tol must be > 0 or None, got {self.tol}")
        if self.seed is not None:
            _require_int("seed", self.seed, minimum=0)  # numpy and Qiskit seeds must be non-negative
        for name in ("gamma_range", "beta_range"):
            rng = getattr(self, name)
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                raise TypeError(f"{name} must be a (low, high) pair, got {rng!r}")
            lo, hi = (_require_real(f"{name}[{i}]", x) for i, x in enumerate(rng))
            if not lo < hi:
                raise ValueError(f"{name} must satisfy low < high, got {rng!r}")
        if self.initial_point is not None:
            if isinstance(self.initial_point, (str, bytes)) or not hasattr(self.initial_point, "__len__"):
                raise TypeError(f"initial_point must be a sequence of 2p numbers, got {self.initial_point!r}")
            if len(self.initial_point) != 2 * self.p:
                raise ValueError(f"initial_point must have length 2p = {2 * self.p}")
            for i, x in enumerate(self.initial_point):
                _require_real(f"initial_point[{i}]", x)
        _require_real("statevector_probability_threshold", self.statevector_probability_threshold)
        if not 0.0 <= self.statevector_probability_threshold < 1.0:
            raise ValueError("statevector_probability_threshold must be in [0, 1)")


def _require_int(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__} {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _require_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__} {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


@dataclass
class OptimizationTrace:
    initial_point: list[float]
    final_point: list[float]
    final_expectation: float
    nfev: int
    success: bool
    message: str
    history: list[float] = field(default_factory=list)  # expected cut per evaluation


def optimize_parameters(qaoa: QAOACircuit, config: QAOAConfig) -> tuple[list[float], list[float], list[OptimizationTrace]]:
    """Maximise the expected cut over ``(gammas, betas)``.

    Returns ``(best_gammas, best_betas, traces)`` with one trace per restart.
    """
    config.validate()
    p = qaoa.p
    rng = np.random.default_rng(config.seed)
    cut_values = all_cut_values(qaoa.graph)

    def expectation(x: np.ndarray) -> float:
        g, b = x[:p], x[p:]
        if config.objective == "statevector":
            return statevector_expectation(qaoa, g, b, cut_values=cut_values)
        counts = sample_counts(qaoa, g, b, shots=config.shots, seed=config.seed)
        return sum(
            c * cut_values[assignment_to_index(bitstring_to_assignment(k, qaoa.graph.num_nodes, order="qiskit"))]
            for k, c in counts.items()
        ) / config.shots

    traces: list[OptimizationTrace] = []
    for r in range(config.n_restarts):
        if r == 0 and config.initial_point is not None:
            x0 = np.asarray(config.initial_point, dtype=float)
        else:
            # Random starting point drawn from the INITIALISATION ranges (not bounds).
            x0 = np.concatenate([rng.uniform(*config.gamma_range, p), rng.uniform(*config.beta_range, p)])
        history: list[float] = []

        def objective(x: np.ndarray) -> float:
            val = expectation(x)
            history.append(val)
            return -val  # scipy minimises

        # Unconstrained: no bounds are passed, so angles may leave the initialisation ranges.
        res = minimize(objective, x0, method=config.optimizer, tol=config.tol, options={"maxiter": config.maxiter})
        traces.append(
            OptimizationTrace(
                initial_point=[float(v) for v in x0],
                final_point=[float(v) for v in res.x],
                final_expectation=float(-res.fun),
                nfev=int(res.nfev),
                success=bool(res.success),
                message=str(res.message),
                history=history,
            )
        )
    best = max(traces, key=lambda t: t.final_expectation)
    return best.final_point[:p], best.final_point[p:], traces


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@dataclass
class QAOAResult:
    """Structured logical (noiseless) QAOA result for integration/metrics.

    Four distinct kinds of quantity -- do not mix them up:

    * Exact classical reference (deterministic): ``exact``, ``optimal_value``.
      ``None`` when the graph exceeded ``max_exact_nodes``.
    * Statevector-derived (deterministic for fixed angles, up to floating point):
      ``expected_cut``, ``expected_approximation_ratio``,
      ``optimal_solution_probability``, ``best_cut_with_statevector_probability``
      (max cut among states with probability > ``statevector_probability_threshold``).
    * Finite-shot samples (stochastic, seeded by ``seed``): ``counts``,
      ``cut_distribution``, ``sampled_mean_cut``, ``sampled_approximation_ratio``,
      ``best_observed_*``, ``most_frequent_bitstring``, ``sampled_optimal_frequency``
      (fraction of shots whose cut equals the exact optimum -- the finite-shot
      success frequency, an estimate of ``optimal_solution_probability``, not the
      same quantity). The best observed sample is NOT the expected value.
    * Optimiser status: ``optimizer_success`` is SciPy's termination flag for the
      best restart (details in ``optimizer``). It says the optimiser stopped
      normally, NOT that the Max-Cut optimum was found.

    ``best_observed_is_exact_optimum`` is an explicit comparison
    ``|best_observed_cut - optimal_value| <= CUT_ATOL``; ``None`` when no exact
    reference is available. Approximation ratios are ``None`` when the exact
    optimum is 0 or unavailable.
    """

    graph: dict[str, Any]
    p: int
    gammas: list[float]
    betas: list[float]
    exact: dict[str, Any] | None
    optimal_value: float | None
    expected_cut: float
    expected_approximation_ratio: float | None
    optimal_solution_probability: float | None
    best_cut_with_statevector_probability: float | None
    statevector_probability_threshold: float
    shots: int
    seed: int | None
    counts: dict[str, int]
    cut_distribution: dict[float, int]
    sampled_mean_cut: float
    sampled_approximation_ratio: float | None
    best_observed_cut: float
    best_observed_bitstring: str
    best_observed_assignment: list[int]
    best_observed_partition: list[list[int]]
    best_observed_ratio: float | None
    best_observed_is_exact_optimum: bool | None
    sampled_optimal_frequency: float | None
    most_frequent_bitstring: str
    bit_order: str
    evaluation_mode: str
    optimizer_success: bool
    optimizer: dict[str, Any]
    circuit: dict[str, Any]
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cut_distribution"] = {str(k): v for k, v in self.cut_distribution.items()}
        return d


def run_qaoa(
    graph: WeightedGraph,
    config: QAOAConfig | None = None,
    *,
    exact: ExactSolution | None = None,
    max_exact_nodes: int = DEFAULT_MAX_EXACT_NODES,
) -> QAOAResult:
    """Optimise QAOA angles, sample the logical circuit, and compare with the exact optimum.

    The exact reference is computed automatically when ``graph.num_nodes <= max_exact_nodes``.
    """
    config = config or QAOAConfig()
    config.validate()
    t0 = time.perf_counter()

    if exact is None and graph.num_nodes <= max_exact_nodes:
        exact = solve_exact(graph, max_nodes=max_exact_nodes)
    opt = exact.optimal_value if exact is not None else None

    qaoa = build_qaoa_circuit(graph, config.p, measure=True)
    gammas, betas, traces = optimize_parameters(qaoa, config)

    probs = statevector_probabilities(qaoa, gammas, betas)
    cut_values = all_cut_values(graph)
    expected = float(probs @ cut_values)
    p_opt = None
    if exact is not None:
        p_opt = float(probs[np.isclose(cut_values, opt, rtol=0.0, atol=CUT_ATOL)].sum())
    sv_best = best_cut_with_statevector_probability(probs, cut_values, config.statevector_probability_threshold)

    counts = sample_counts(qaoa, gammas, betas, shots=config.shots, seed=config.seed)
    samples = decode_counts(graph, counts)
    best = samples[0]
    sampled_mean = sum(s.cut_value * s.count for s in samples) / config.shots
    most_frequent = max(samples, key=lambda s: (s.count, s.cut_value)).bitstring

    def ratio(v: float) -> float | None:
        return approximation_ratio(v, opt) if opt is not None else None

    best_trace = max(traces, key=lambda t: t.final_expectation)
    return QAOAResult(
        graph=graph.to_dict(),
        p=config.p,
        gammas=gammas,
        betas=betas,
        exact=exact.to_dict() if exact is not None else None,
        optimal_value=opt,
        expected_cut=expected,
        expected_approximation_ratio=ratio(expected),
        optimal_solution_probability=p_opt,
        best_cut_with_statevector_probability=sv_best,
        statevector_probability_threshold=config.statevector_probability_threshold,
        shots=config.shots,
        seed=config.seed,
        counts=dict(sorted(counts.items())),
        cut_distribution=cut_distribution(samples),
        sampled_mean_cut=sampled_mean,
        sampled_approximation_ratio=ratio(sampled_mean),
        best_observed_cut=best.cut_value,
        best_observed_bitstring=best.bitstring,
        best_observed_assignment=list(best.assignment),
        best_observed_partition=[list(s) for s in best.partition],
        best_observed_ratio=ratio(best.cut_value),
        best_observed_is_exact_optimum=(None if opt is None else abs(best.cut_value - opt) <= CUT_ATOL),
        sampled_optimal_frequency=(None if opt is None else
                                   sum(s.count for s in samples if abs(s.cut_value - opt) <= CUT_ATOL) / config.shots),
        most_frequent_bitstring=most_frequent,
        bit_order=BIT_ORDER,
        evaluation_mode=f"optimise:{config.objective}; report:statevector expectation + {config.shots} noiseless shots",
        optimizer_success=best_trace.success,
        optimizer={
            "name": config.optimizer,
            "maxiter": config.maxiter,
            "tol": config.tol,
            "n_restarts": config.n_restarts,
            "objective": config.objective,
            "seed": config.seed,
            "best_restart": traces.index(best_trace),
            "restarts": [asdict(t) for t in traces],
        },
        circuit=qaoa.metadata(),
        runtime_seconds=time.perf_counter() - t0,
    )


def logical_circuit_for_routing(
    graph: WeightedGraph, p: int, gammas: Sequence[float] | None = None, betas: Sequence[float] | None = None
) -> QuantumCircuit:
    """Convenience for Member 4: the measured logical circuit, bound if angles are given.

    Build once and pass the *same* object to the Processor A and Processor B
    compilation runs so both start from an identical logical circuit.
    """
    qaoa = build_qaoa_circuit(graph, p, measure=True)
    if gammas is None and betas is None:
        return qaoa.circuit
    if gammas is None or betas is None:
        raise ValueError("provide both gammas and betas, or neither")
    return qaoa.bind(gammas, betas)
