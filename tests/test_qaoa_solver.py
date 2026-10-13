"""Correctness tests for src/qaoa_solver.py (Member 3).

All graphs here are SYNTHETIC test instances. Sign/convention checks compare the
Qiskit circuit against an independent numpy/scipy construction of
exp(-i gamma C_hat) and exp(-i beta sum X), so they would fail if the RZZ angle
sign or factor were wrong.
"""

import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from src.problem import (
    WeightedGraph,
    all_cut_values,
    assignment_to_index,
    cut_value,
    synthetic_weighted_graph,
)
from src.qaoa_solver import (
    CREG_NAME,
    QAOAConfig,
    build_qaoa_circuit,
    cost_hamiltonian,
    cut_distribution,
    decode_counts,
    logical_circuit_for_routing,
    optimize_parameters,
    run_qaoa,
    sample_counts,
    statevector_expectation,
    statevector_probabilities,
)

# A path 0-1-2 with distinct weights: asymmetric under vertex reversal.
PATH = WeightedGraph.from_edges(3, [(0, 1, 1.0), (1, 2, 3.0)], name="synthetic_path3", source="synthetic")


def reference_state(graph, gammas, betas):
    """Independent numpy QAOA state: prod U_B(beta) U_C(gamma) |+>^n, little-endian basis."""
    n = graph.num_nodes
    c_diag = all_cut_values(graph)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    b_op = sum(np.kron(np.eye(2 ** (n - 1 - j)), np.kron(x, np.eye(2**j))) for j in range(n))
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * c_diag) * psi
        psi = expm(-1j * b * b_op) @ psi
    return psi


# -- Hamiltonian --------------------------------------------------------------


def test_hamiltonian_diagonal_equals_classical_cut():
    for g in (synthetic_weighted_graph(), PATH):
        h = cost_hamiltonian(g).to_matrix()
        assert np.allclose(h, np.diag(np.diag(h)))  # diagonal
        assert np.allclose(np.diag(h).real, all_cut_values(g))


def test_hamiltonian_basis_expectation_matches_cut_value():
    g = synthetic_weighted_graph()
    h = cost_hamiltonian(g).to_matrix()
    for x in [(0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 1, 0)]:
        k = assignment_to_index(x)
        assert h[k, k].real == pytest.approx(cut_value(g, x))


# -- circuit structure ------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3])
def test_circuit_width_gate_counts_and_parameters(p):
    g = synthetic_weighted_graph()
    q = build_qaoa_circuit(g, p)
    ops = q.circuit.count_ops()
    assert q.num_qubits == 4 and q.circuit.num_clbits == 4
    assert ops["h"] == 4
    assert ops["rzz"] == p * g.num_edges
    assert ops["rx"] == p * 4
    assert ops["measure"] == 4
    assert len(q.circuit.parameters) == 2 * p
    assert {prm.name for prm in q.circuit.parameters} == {f"gamma[{l}]" for l in range(p)} | {f"beta[{l}]" for l in range(p)}


def test_depth_grows_with_p():
    g = synthetic_weighted_graph()
    depths = [build_qaoa_circuit(g, p, measure=False).circuit.depth() for p in (1, 2, 3)]
    assert depths[0] < depths[1] < depths[2]


def test_unmeasured_circuit_has_no_clbits():
    q = build_qaoa_circuit(PATH, 1, measure=False)
    assert q.circuit.num_clbits == 0 and "measure" not in q.circuit.count_ops()


def test_cost_layer_has_one_rzz_per_edge_with_weighted_angle():
    g = synthetic_weighted_graph()
    q = build_qaoa_circuit(g, 1, measure=False)
    gamma = 0.37
    bound = q.bind([gamma], [0.0])
    rzz = [(tuple(bound.find_bit(b).index for b in inst.qubits), float(inst.operation.params[0]))
           for inst in bound.data if inst.operation.name == "rzz"]
    expected = {(u, v): -w * gamma for u, v, w in g.edges}
    assert len(rzz) == len(expected)
    assert {tuple(sorted(qs)): ang for qs, ang in rzz} == pytest.approx(expected)


def test_mixer_covers_every_qubit_each_layer():
    q = build_qaoa_circuit(synthetic_weighted_graph(), 2)
    rx_qubits = [q.circuit.find_bit(inst.qubits[0]).index for inst in q.circuit.data if inst.operation.name == "rx"]
    assert sorted(rx_qubits) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_zero_weight_edges_emit_no_gate():
    g = WeightedGraph.from_edges(3, [(0, 1, 0.0), (1, 2, 2.0)])
    assert build_qaoa_circuit(g, 1).circuit.count_ops()["rzz"] == 1


def test_edgeless_graph_still_binds():
    g = WeightedGraph.from_edges(2, [])
    q = build_qaoa_circuit(g, 1)
    assert q.bind([0.3], [0.2]).num_parameters == 0


def test_measurements_map_qubit_i_to_clbit_i():
    q = build_qaoa_circuit(synthetic_weighted_graph(), 1)
    pairs = [(q.circuit.find_bit(i.qubits[0]).index, q.circuit.find_bit(i.clbits[0]).index)
             for i in q.circuit.data if i.operation.name == "measure"]
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert q.vertex_to_qubit == (0, 1, 2, 3) and q.qubit_to_clbit == (0, 1, 2, 3)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_depth_rejected(bad):
    with pytest.raises(ValueError):
        build_qaoa_circuit(PATH, bad)


def test_bind_validates_angle_lengths():
    q = build_qaoa_circuit(PATH, 2)
    with pytest.raises(ValueError):
        q.bind([0.1], [0.2, 0.3])
    with pytest.raises(ValueError):
        q.bind([0.1, math.nan], [0.2, 0.3])
    assert q.bind([0.1, 0.2], [0.3, 0.4]).num_parameters == 0


def test_logical_circuit_contains_no_topology_specific_ops():
    ops = set(build_qaoa_circuit(synthetic_weighted_graph(), 2).circuit.count_ops())
    assert ops == {"h", "rzz", "rx", "measure"}  # no swap / no routing artefacts


# -- sign / angle convention ----------------------------------------------------


def test_cost_unitary_equals_exp_minus_i_gamma_C():
    g = synthetic_weighted_graph()
    gamma = 0.41
    q = build_qaoa_circuit(g, 1, measure=False)
    # Isolate the cost layer by setting beta=0 and removing the initial Hadamards.
    circ = q.bind([gamma], [0.0])
    circ.data = [i for i in circ.data if i.operation.name == "rzz"]
    target = np.diag(np.exp(-1j * gamma * all_cut_values(g)))
    assert Operator(circ).equiv(Operator(target))
    # The opposite sign is a genuinely different unitary (not just a global phase).
    assert not Operator(circ).equiv(Operator(target.conj()))


def test_full_state_matches_independent_reference():
    rng = np.random.default_rng(3)
    for g in (synthetic_weighted_graph(), PATH):
        for p in (1, 2):
            gammas, betas = rng.uniform(-2, 2, p), rng.uniform(-2, 2, p)
            probs = statevector_probabilities(build_qaoa_circuit(g, p), gammas, betas)
            ref = np.abs(reference_state(g, gammas, betas)) ** 2
            assert np.allclose(probs, ref, atol=1e-10)


def test_single_edge_p1_closed_form():
    """For one edge of weight w: <C> = w/2 * (1 + sin(4 beta) sin(w gamma)) under our convention.

    Flipping the sign convention of gamma would flip the sign of the sin(w gamma) term.
    """
    w = 2.5
    g = WeightedGraph.from_edges(2, [(0, 1, w)])
    q = build_qaoa_circuit(g, 1)
    for gamma, beta in [(0.3, 0.2), (1.1, -0.4), (-0.7, 0.35)]:
        expected = w / 2 * (1 + math.sin(4 * beta) * math.sin(w * gamma))
        assert statevector_expectation(q, [gamma], [beta]) == pytest.approx(expected, abs=1e-10)


def test_zero_angles_give_uniform_distribution():
    g = synthetic_weighted_graph()
    q = build_qaoa_circuit(g, 1)
    assert np.allclose(statevector_probabilities(q, [0.0], [0.0]), 1 / 16)
    assert statevector_expectation(q, [0.0], [0.0]) == pytest.approx(g.total_weight / 2)


# -- measurement / decoding ------------------------------------------------------------


def test_counts_keys_decode_with_vertex0_rightmost():
    """Sampled counts, decoded with vertex 0 = rightmost char, must match the exact
    statevector probabilities indexed by k = sum_i x_i 2**i.

    PATH has distinct weights, so its QAOA distribution is not symmetric under
    reversing the vertex order; a reversed decoder would fail this test.
    """
    q = build_qaoa_circuit(PATH, 1)
    gammas, betas = [0.6], [0.25]
    probs = statevector_probabilities(q, gammas, betas)
    counts = sample_counts(q, gammas, betas, shots=20000, seed=11)
    for s in decode_counts(PATH, counts):
        k = assignment_to_index(s.assignment)
        assert s.count / 20000 == pytest.approx(probs[k], abs=0.015)
    # This check is not vacuous: the distribution is not reversal-symmetric.
    rev = [int(format(k, "03b")[::-1], 2) for k in range(8)]
    assert not np.allclose(probs, probs[rev], atol=0.02)


def test_decode_counts_reversal_sensitive():
    g = synthetic_weighted_graph()
    samples = decode_counts(g, {"1000": 5, "0001": 3})
    by_key = {s.bitstring: s for s in samples}
    assert by_key["1000"].assignment == (0, 0, 0, 1) and by_key["1000"].cut_value == 7.0
    assert by_key["0001"].assignment == (1, 0, 0, 0) and by_key["0001"].cut_value == 10.0
    assert samples[0].bitstring == "0001"  # sorted by cut value descending
    assert cut_distribution(samples) == {7.0: 5, 10.0: 3}


def test_sampler_register_name():
    q = build_qaoa_circuit(PATH, 1)
    assert q.circuit.cregs[0].name == CREG_NAME
    counts = sample_counts(q, [0.1], [0.1], shots=64, seed=0)
    assert sum(counts.values()) == 64 and all(len(k) == 3 for k in counts)


def test_sampling_is_reproducible_with_seed():
    q = build_qaoa_circuit(PATH, 1)
    assert sample_counts(q, [0.4], [0.3], shots=200, seed=5) == sample_counts(q, [0.4], [0.3], shots=200, seed=5)


# -- optimisation & end-to-end ---------------------------------------------------------


def test_optimizer_improves_on_uniform_baseline():
    g = synthetic_weighted_graph()
    q = build_qaoa_circuit(g, 1)
    gammas, betas, traces = optimize_parameters(q, QAOAConfig(p=1, n_restarts=2, maxiter=100, seed=0))
    assert len(traces) == 2 and all(t.nfev > 0 for t in traces)
    assert statevector_expectation(q, gammas, betas) > g.total_weight / 2 + 1.0


def test_initial_point_is_used():
    q = build_qaoa_circuit(PATH, 1)
    _, _, traces = optimize_parameters(q, QAOAConfig(p=1, n_restarts=1, maxiter=5, initial_point=[0.2, 0.1]))
    assert traces[0].initial_point == [0.2, 0.1]


def test_config_validation():
    with pytest.raises(ValueError):
        QAOAConfig(objective="magic").validate()
    with pytest.raises(ValueError):
        QAOAConfig(p=2, initial_point=[0.1, 0.2]).validate()


def test_run_qaoa_result_consistency():
    g = synthetic_weighted_graph()
    r = run_qaoa(g, QAOAConfig(p=1, n_restarts=2, maxiter=80, shots=1000, seed=42))
    assert r.optimal_value == 11.0
    assert sum(r.counts.values()) == 1000 == sum(r.cut_distribution.values())
    assert 0 <= r.expected_cut <= r.optimal_value
    assert r.expected_approximation_ratio == pytest.approx(r.expected_cut / 11.0)
    assert r.best_observed_cut <= r.optimal_value
    assert r.best_observed_cut == cut_value(g, r.best_observed_assignment)
    assert r.best_observed_cut == max(r.cut_distribution)
    assert 0 < r.optimal_solution_probability <= 1
    assert r.bit_order == "qiskit"
    d = r.to_dict()
    import json
    json.dumps(d)  # fully JSON-serialisable


def test_run_qaoa_shots_objective_mode():
    r = run_qaoa(PATH, QAOAConfig(p=1, objective="shots", n_restarts=1, maxiter=30, shots=256, seed=1))
    assert r.optimal_value == 4.0 and r.expected_cut > 0


def test_run_qaoa_zero_optimum_ratio_undefined():
    r = run_qaoa(WeightedGraph.from_edges(2, []), QAOAConfig(p=1, n_restarts=1, maxiter=5, shots=10))
    assert r.optimal_value == 0.0
    assert r.expected_approximation_ratio is None and r.best_observed_ratio is None


def test_counts_stay_in_logical_order_after_routing():
    """Contract for Member 4: transpiling/routing must not change how counts decode.

    Uses a GENERIC test-only 5-qubit line coupling map (NOT Processor A or B) that
    forces SWAP insertion; the measured distribution must still match the logical
    statevector probabilities indexed by vertex.
    """
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    g = synthetic_weighted_graph()
    q = build_qaoa_circuit(g, 1)
    gammas, betas = [0.5], [0.3]
    probs = statevector_probabilities(q, gammas, betas)
    compiled = transpile(q.bind(gammas, betas), coupling_map=CouplingMap.from_line(5),
                         basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=7)
    assert compiled.num_qubits == 5
    from qiskit.primitives import StatevectorSampler
    counts = getattr(StatevectorSampler(seed=3).run([compiled], shots=20000).result()[0].data, CREG_NAME).get_counts()
    for s in decode_counts(g, counts):
        assert s.count / 20000 == pytest.approx(probs[assignment_to_index(s.assignment)], abs=0.015)


def test_logical_circuit_for_routing():
    g = synthetic_weighted_graph()
    unbound = logical_circuit_for_routing(g, 2)
    assert unbound.num_parameters == 4
    bound = logical_circuit_for_routing(g, 2, [0.1, 0.2], [0.3, 0.4])
    assert bound.num_parameters == 0
    with pytest.raises(ValueError):
        logical_circuit_for_routing(g, 2, [0.1, 0.2])


# -- strict QAOAConfig validation (audit fix) -------------------------------------
# Before the fix, shots=True, maxiter=2.5, n_restarts=True, seed=True and tol<=0/NaN were
# silently accepted, and NaN initial points / reversed ranges failed late with obscure errors.


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(p=0), ValueError), (dict(p=True), TypeError), (dict(p=1.0), TypeError),
        (dict(maxiter=2.5), TypeError), (dict(maxiter=0), ValueError), (dict(maxiter=True), TypeError),
        (dict(n_restarts=True), TypeError), (dict(n_restarts=0), ValueError),
        (dict(shots=True), TypeError), (dict(shots=0), ValueError), (dict(shots=10.0), TypeError),
        (dict(tol=-1.0), ValueError), (dict(tol=0.0), ValueError), (dict(tol=math.nan), ValueError),
        (dict(tol="1e-6"), TypeError),
        (dict(seed=True), TypeError), (dict(seed=1.5), TypeError), (dict(seed=-1), ValueError),
        (dict(optimizer=""), TypeError), (dict(optimizer=None), TypeError),
        (dict(objective="magic"), ValueError),
        (dict(gamma_range=(1.0, 0.0)), ValueError), (dict(gamma_range=(0.0, math.inf)), ValueError),
        (dict(beta_range=(math.nan, 1.0)), ValueError), (dict(beta_range=(0.0,)), TypeError),
        (dict(gamma_range=(0, True)), TypeError),
        (dict(initial_point=[math.nan, 0.1]), ValueError), (dict(initial_point=[math.inf, 0.1]), ValueError),
        (dict(initial_point=[0.1, "0.2"]), TypeError), (dict(initial_point=[0.1]), ValueError),
        (dict(initial_point=0.1), TypeError),
        (dict(statevector_probability_threshold=math.nan), ValueError),
        (dict(statevector_probability_threshold=-0.1), ValueError),
        (dict(statevector_probability_threshold=1.0), ValueError),
    ],
)
def test_config_rejects_invalid_settings(kwargs, exc):
    with pytest.raises(exc):
        QAOAConfig(**kwargs).validate()


def test_config_defaults_and_valid_variants_pass():
    QAOAConfig().validate()
    QAOAConfig(p=2, tol=None, seed=None, initial_point=[0.1, 0.2, 0.3, 0.4], gamma_range=(-1, 1)).validate()
    QAOAConfig(seed=np.int64(3), shots=np.int64(10)).validate()


def test_run_qaoa_validates_before_doing_work():
    with pytest.raises(TypeError):
        run_qaoa(PATH, QAOAConfig(shots=True))
