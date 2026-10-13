"""Tests for routing, capacity validation, and processor compilation (Member 4)."""

import pytest

from src.problem import synthetic_weighted_graph
from src.processors import load_processor
from src.routing import (
    CapacityError,
    compile_for_processor,
    compare_processors,
    validate_capacity,
)


def test_capacity_validation_rejects_oversized_circuit():
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        pytest.skip("Qiskit not installed in test environment")

    # 6-qubit circuit onto 5-qubit Processor A
    qc = QuantumCircuit(6)
    config_a = load_processor("processor_A")
    with pytest.raises(CapacityError, match="requires 6 logical qubits"):
        validate_capacity(qc, config_a)


def test_capacity_validation_accepts_fitting_circuit():
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        pytest.skip("Qiskit not installed in test environment")

    # 4-qubit circuit onto 5-qubit Processor A
    qc = QuantumCircuit(4)
    config_a = load_processor("processor_A")
    validate_capacity(qc, config_a)  # Should not raise


def test_compilation_reproducibility():
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        pytest.skip("Qiskit not installed in test environment")

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    res1 = compile_for_processor(qc, "processor_A", seed_transpiler=31)
    res2 = compile_for_processor(qc, "processor_A", seed_transpiler=31)

    assert res1.compiled_depth == res2.compiled_depth
    assert res1.compiled_cx_count == res2.compiled_cx_count
    assert res1.compiled_total_2q_gates == res2.compiled_total_2q_gates


def test_compare_processors_handles_capacity_honestly():
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        pytest.skip("Qiskit not installed in test environment")

    qc = QuantumCircuit(6)  # Fits on B (9q), fails on A (5q)
    results = compare_processors(qc)

    assert results["processor_A"]["status"] == "CAPACITY_EXCEEDED"
    assert results["processor_B"]["status"] == "SUCCESS"
    assert results["processor_B"]["metrics"]["num_physical_qubits"] == 9
