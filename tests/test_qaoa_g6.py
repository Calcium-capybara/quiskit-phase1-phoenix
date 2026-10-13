"""QAOA hardening tests on the team-defined G6 graph (6 vertices, 9 edges, optimum 24).

Covers circuit construction, bit order (with strings whose reversal is NOT the
complement), initialisation ranges vs optimiser bounds, finite-shot
reproducibility, and result-field semantics. G6 is team-defined, not official.
"""

import json
import math

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from scipy.optimize import OptimizeResult

import src.qaoa_solver as qs
from src.problem import all_cut_values, assignment_to_index, cut_value_from_bitstring, load_graph
from src.qaoa_solver import (
    CREG_NAME,
    QAOAConfig,
    best_cut_with_statevector_probability,
    build_qaoa_circuit,
    decode_counts,
    logical_circuit_for_routing,
    optimize_parameters,
    run_qaoa,
    sample_counts,
    statevector_probabilities,
)

G6 = load_graph()
FAST = dict(n_restarts=1, maxiter=60, shots=1024, seed=7)


def ops_named(circ, name):
    return [i for i in circ.data if i.operation.name == name]


def qubits_of(circ, inst):
    return tuple(circ.find_bit(q).index for q in inst.qubits)


# -- circuit construction ------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3])
def test_g6_circuit_width_and_layer_counts(p):
    q = build_qaoa_circuit(G6, p)
    ops = q.circuit.count_ops()
    assert q.num_qubits == 6 and q.circuit.num_clbits == 6
    assert ops["h"] == 6 and ops["rzz"] == 9 * p and ops["rx"] == 6 * p and ops["measure"] == 6
    assert len(q.circuit.parameters) == 2 * p


def test_g6_initial_hadamard_layer_comes_first():
    circ = build_qaoa_circuit(G6, 1).circuit
    first = circ.data[:6]
    assert [i.operation.name for i in first] == ["h"] * 6
    assert sorted(qubits_of(circ, i)[0] for i in first) == list(range(6))


def test_g6_cost_and_mixer_angles_per_layer():
    gammas, betas = [0.3, 0.7], [0.11, 0.42]
    circ = build_qaoa_circuit(G6, 2, measure=False).bind(gammas, betas)
    rzz, rx = ops_named(circ, "rzz"), ops_named(circ, "rx")
    for layer in range(2):
        got = {tuple(sorted(qubits_of(circ, i))): float(i.operation.params[0]) for i in rzz[9 * layer: 9 * layer + 9]}
        want = {(u, v): -w * gammas[layer] for u, v, w in G6.edges}  # RZZ(-w*gamma) convention
        assert got == pytest.approx(want)
        layer_rx = rx[6 * layer: 6 * layer + 6]
        assert sorted(qubits_of(circ, i)[0] for i in layer_rx) == list(range(6))
        assert all(float(i.operation.params[0]) == pytest.approx(2 * betas[layer]) for i in layer_rx)


def test_g6_cost_layer_is_exp_minus_i_gamma_C():
    gamma = 0.37
    circ = build_qaoa_circuit(G6, 1, measure=False).bind([gamma], [0.0])
    circ.data = ops_named(circ, "rzz")
    target = np.diag(np.exp(-1j * gamma * all_cut_values(G6)))
    assert Operator(circ).equiv(Operator(target))
    assert not Operator(circ).equiv(Operator(target.conj()))


def test_g6_statevector_circuit_has_no_measurements():
    q = build_qaoa_circuit(G6, 2, measure=False)
    assert q.circuit.num_clbits == 0 and not ops_named(q.circuit, "measure")


def test_g6_sampling_circuit_measures_qubit_i_into_clbit_i():
    circ = build_qaoa_circuit(G6, 1).circuit
    pairs = [(circ.find_bit(i.qubits[0]).index, circ.find_bit(i.clbits[0]).index) for i in ops_named(circ, "measure")]
    assert pairs == [(i, i) for i in range(6)]
    assert circ.cregs[0].name == CREG_NAME


@pytest.mark.parametrize("p", [0, -1, 1.5, True, None])
def test_invalid_depth(p):
    with pytest.raises(ValueError):
        build_qaoa_circuit(G6, p)


@pytest.mark.parametrize("gammas,betas", [([0.1], [0.2, 0.3]), ([0.1, 0.2], [0.3]), ([math.nan, 0.1], [0.1, 0.1]),
                                          ([math.inf, 0.1], [0.1, 0.1])])
def test_invalid_parameters(gammas, betas):
    with pytest.raises(ValueError):
        build_qaoa_circuit(G6, 2).bind(gammas, betas)


def test_g6_downstream_compilation_compatibility():
    """The logical circuit transpiles with Qiskit; measurements survive. No processor data used."""
    circ = logical_circuit_for_routing(G6, 1, [0.4], [0.2])
    assert circ.num_parameters == 0
    compiled = transpile(circ, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=3)
    assert compiled.num_qubits == 6 and compiled.count_ops()["measure"] == 6


# -- bit order (reversal is NOT the complement for these strings) -------------------


@pytest.mark.parametrize(
    "key,qiskit_cut,vertex_cut",
    [("000011", 13.0, 9.0),   # vertices {0,1}: 2+4+5+2 = 13; misread {4,5}: 4+1+4 = 9
     ("000100", 11.0, 9.0)],  # vertex {2}: 2+5+3+1 = 11; misread {3}: 2+3+4 = 9
)
def test_g6_bit_order_detects_reversal(key, qiskit_cut, vertex_cut):
    (sample,) = decode_counts(G6, {key: 1})
    assert sample.cut_value == qiskit_cut
    assert cut_value_from_bitstring(G6, key, order="vertex") == vertex_cut


def test_g6_counts_after_routing_decode_in_logical_order():
    """Generic TEST-ONLY 6-qubit line map (not Processor A/B) forces SWAPs."""
    gammas, betas = [0.5], [0.3]
    probs = statevector_probabilities(build_qaoa_circuit(G6, 1), gammas, betas)
    rev = [int(format(k, "06b")[::-1], 2) for k in range(64)]
    assert not np.allclose(probs, probs[rev], atol=0.01)  # distribution is reversal-sensitive
    compiled = transpile(logical_circuit_for_routing(G6, 1, gammas, betas), coupling_map=CouplingMap.from_line(6),
                         basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=5)
    from qiskit.primitives import StatevectorSampler
    counts = getattr(StatevectorSampler(seed=9).run([compiled], shots=40000).result()[0].data, CREG_NAME).get_counts()
    for s in decode_counts(G6, counts):
        assert s.count / 40000 == pytest.approx(probs[assignment_to_index(s.assignment)], abs=0.01)


# -- initialisation ranges vs optimiser bounds ----------------------------------------


def test_optimizer_is_called_without_bounds(monkeypatch):
    calls = []
    real = qs.minimize

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(qs, "minimize", spy)
    optimize_parameters(build_qaoa_circuit(G6, 1), QAOAConfig(p=1, n_restarts=2, maxiter=4, seed=1))
    assert len(calls) == 2
    assert all(c["method"] == "COBYLA" and "bounds" not in c and "constraints" not in c for c in calls)


def test_random_initial_points_lie_in_initialisation_ranges():
    cfg = QAOAConfig(p=2, n_restarts=4, maxiter=6, seed=3)  # COBYLA needs maxiter >= 2p + 2
    _, _, traces = optimize_parameters(build_qaoa_circuit(G6, 2), cfg)
    for t in traces:
        g, b = t.initial_point[:2], t.initial_point[2:]
        assert all(cfg.gamma_range[0] <= x < cfg.gamma_range[1] for x in g)
        assert all(cfg.beta_range[0] <= x < cfg.beta_range[1] for x in b)


def test_initial_point_outside_ranges_is_not_clipped():
    _, _, traces = optimize_parameters(build_qaoa_circuit(G6, 1),
                                       QAOAConfig(p=1, n_restarts=1, maxiter=4, initial_point=[5.0, -1.0]))
    assert traces[0].initial_point == [5.0, -1.0]


# -- finite-shot reproducibility -------------------------------------------------------


def test_sampler_receives_the_seed(monkeypatch):
    seen = []
    real = qs.StatevectorSampler

    def factory(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(qs, "StatevectorSampler", factory)
    sample_counts(build_qaoa_circuit(G6, 1), [0.3], [0.2], shots=16, seed=4242)
    assert seen == [4242]


def test_same_seed_same_counts():
    q = build_qaoa_circuit(G6, 1)
    a = sample_counts(q, [0.3], [0.2], shots=500, seed=11)
    b = sample_counts(q, [0.3], [0.2], shots=500, seed=11)
    assert a == b and sum(a.values()) == 500


def test_statevector_is_deterministic():
    q = build_qaoa_circuit(G6, 2)
    assert np.array_equal(statevector_probabilities(q, [0.3, 0.1], [0.2, 0.4]),
                          statevector_probabilities(q, [0.3, 0.1], [0.2, 0.4]))


def test_run_qaoa_reproducible_with_same_seed_and_config():
    keys = ["gammas", "betas", "expected_cut", "counts", "sampled_mean_cut", "best_observed_cut",
            "best_observed_bitstring", "most_frequent_bitstring", "optimizer_success"]
    r1 = run_qaoa(G6, QAOAConfig(p=1, **FAST)).to_dict()
    r2 = run_qaoa(G6, QAOAConfig(p=1, **FAST)).to_dict()
    assert {k: r1[k] for k in keys} == {k: r2[k] for k in keys}


# -- statevector-derived metric ------------------------------------------------------------


def test_statevector_metric_threshold_rule():
    cuts = np.array([1.0, 5.0, 3.0])
    thr = 1e-12
    assert best_cut_with_statevector_probability(np.array([0.6, 1e-13, 0.4]), cuts, thr) == 3.0  # below: excluded
    assert best_cut_with_statevector_probability(np.array([0.6, 2e-12, 0.4]), cuts, thr) == 5.0  # above: included
    assert best_cut_with_statevector_probability(np.array([0.6, thr, 0.4]), cuts, thr) == 3.0    # equal: excluded (strict >)
    assert best_cut_with_statevector_probability(np.zeros(3), cuts, thr) is None                  # none qualifies
    with pytest.raises(ValueError):
        best_cut_with_statevector_probability(np.ones(2), cuts, thr)


def test_statevector_metric_in_result_uses_configured_threshold():
    cfg = QAOAConfig(p=1, statevector_probability_threshold=1e-6, **FAST)
    r = run_qaoa(G6, cfg)
    probs = statevector_probabilities(build_qaoa_circuit(G6, 1), r.gammas, r.betas)
    assert r.statevector_probability_threshold == 1e-6
    assert r.best_cut_with_statevector_probability == best_cut_with_statevector_probability(
        probs, all_cut_values(G6), 1e-6)
    assert r.best_cut_with_statevector_probability <= r.optimal_value


@pytest.mark.parametrize("thr", [-1e-12, 1.0, 2.0])
def test_invalid_threshold_rejected(thr):
    with pytest.raises(ValueError):
        QAOAConfig(statevector_probability_threshold=thr).validate()


# -- result semantics: optimiser status vs global optimality -----------------------------


def fake_minimize(success):
    """Stand-in optimiser: returns the start point with a chosen termination flag."""
    def _minimize(fun, x0, **kwargs):
        x = np.zeros_like(x0)  # gamma = beta = 0 -> uniform distribution, far from optimal
        return OptimizeResult(x=x, fun=fun(x), success=success, nfev=1, message="fake")
    return _minimize


def test_optimizer_success_does_not_imply_optimum(monkeypatch):
    monkeypatch.setattr(qs, "minimize", fake_minimize(True))
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=1, shots=1, seed=5))
    assert r.optimizer_success is True
    assert r.expected_cut == pytest.approx(29 / 2)               # uniform state: half the total weight
    assert r.expected_approximation_ratio == pytest.approx(14.5 / 24)
    assert r.optimal_solution_probability == pytest.approx(2 / 64)
    assert r.best_observed_is_exact_optimum == (r.best_observed_cut == 24.0)


def test_optimum_can_be_observed_despite_failed_termination(monkeypatch):
    monkeypatch.setattr(qs, "minimize", fake_minimize(False))
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=1, shots=2048, seed=5))
    assert r.optimizer_success is False
    assert r.best_observed_cut == 24.0 and r.best_observed_is_exact_optimum is True


def test_optimizer_success_mirrors_best_restart():
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=2, maxiter=40, shots=64, seed=2))
    assert r.optimizer_success == r.optimizer["restarts"][r.optimizer["best_restart"]]["success"]


def test_unavailable_exact_reference_is_reported_as_none():
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=1, maxiter=10, shots=64, seed=1), max_exact_nodes=3)
    assert r.exact is None and r.optimal_value is None
    assert r.best_observed_is_exact_optimum is None
    assert r.expected_approximation_ratio is None and r.best_observed_ratio is None
    assert r.optimal_solution_probability is None


def test_result_quantities_are_distinct_and_serialisable():
    r = run_qaoa(G6, QAOAConfig(p=1, **FAST))
    d = json.loads(json.dumps(r.to_dict()))
    assert d["optimal_value"] == 24.0 and d["exact"]["optimal_value"] == 24.0
    assert 0 < d["expected_cut"] < 24.0                        # statevector expectation, not a sample
    assert sum(d["counts"].values()) == d["shots"] == 1024     # finite-shot data
    assert d["sampled_mean_cut"] == pytest.approx(
        sum(float(k) * v for k, v in d["cut_distribution"].items()) / 1024)
    assert d["seed"] == 7 and d["bit_order"] == "qiskit"
    for field in ("optimizer_success", "best_observed_is_exact_optimum", "best_cut_with_statevector_probability",
                  "statevector_probability_threshold"):
        assert field in d
    assert "best_cut_with_nonzero_probability" not in d          # ambiguous legacy name not used


# -- statevector metric is not proof of optimisation (audit fix) -------------------


@pytest.mark.parametrize("thr", [math.nan, math.inf, -1e-12, 1.0])
def test_metric_helper_rejects_invalid_threshold(thr):
    with pytest.raises(ValueError):
        best_cut_with_statevector_probability(np.array([0.5, 0.5]), np.array([1.0, 2.0]), thr)


@pytest.mark.parametrize("probs", [[np.nan, 1.0], [-0.1, 1.1], [np.inf, 0.0]])
def test_metric_helper_rejects_invalid_probabilities(probs):
    with pytest.raises(ValueError):
        best_cut_with_statevector_probability(np.array(probs), np.array([9.0, 2.0]), 1e-12)


def test_statevector_support_equals_optimum_even_for_uniform_state():
    """gamma = beta = 0 (no optimisation at all) already 'supports' the optimum 24."""
    probs = statevector_probabilities(build_qaoa_circuit(G6, 1), [0.0], [0.0])
    assert best_cut_with_statevector_probability(probs, all_cut_values(G6), 1e-12) == 24.0
    assert probs[np.isclose(all_cut_values(G6), 24.0)].sum() == pytest.approx(2 / 64)  # yet P(optimal) is tiny


# -- Challenge Kit settings and success frequency ------------------------------------


def test_default_shots_follow_challenge_kit():
    assert QAOAConfig().shots == 4096


def test_sampled_optimal_frequency_is_finite_shot_success_rate():
    r = run_qaoa(G6, QAOAConfig(p=1, **FAST))
    hits = sum(s.count for s in decode_counts(G6, r.counts) if s.cut_value == 24.0)
    assert r.sampled_optimal_frequency == hits / r.shots                  # counted from samples
    assert r.sampled_optimal_frequency == pytest.approx(r.optimal_solution_probability, abs=0.05)  # estimates it
    assert r.best_observed_is_exact_optimum == (hits > 0)


def test_sampled_optimal_frequency_differs_from_statevector_probability(monkeypatch):
    """Uniform state: P(optimal) = 2/64 exactly; 64 shots give a different, sampled frequency."""
    monkeypatch.setattr(qs, "minimize", fake_minimize(True))
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=1, shots=64, seed=5))
    assert r.optimal_solution_probability == pytest.approx(2 / 64)
    assert r.sampled_optimal_frequency * 64 == int(r.sampled_optimal_frequency * 64)  # a count / shots


def test_sampled_optimal_frequency_none_without_exact_reference():
    r = run_qaoa(G6, QAOAConfig(p=1, n_restarts=1, maxiter=10, shots=64, seed=1), max_exact_nodes=3)
    assert r.sampled_optimal_frequency is None
