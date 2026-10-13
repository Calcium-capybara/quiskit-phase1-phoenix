"""Processor-aware transpilation and routing engine (Member 4 takeover).

Integrates the QAOA logical circuit with physical target hardware topologies
(Processor A: 5-qubit linear chain, Processor B: 9-qubit heavy-hex).
Implements capacity validation, layout tracking, SWAP routing measurement,
and architecture comparison metrics.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from src.processors import get_coupling_map, load_processor


class CapacityError(ValueError):
    """Raised when a circuit requires more logical qubits than the physical processor provides."""
    pass


class ConnectivityError(ValueError):
    """Raised when a circuit cannot be satisfied on the processor connectivity map."""
    pass


@dataclass(frozen=True)
class TranspilationResult:
    """Detailed record of transpilation onto a specific quantum processor."""
    processor_name: str
    num_physical_qubits: int
    num_logical_qubits: int
    logical_depth: int
    logical_cx_or_2q_gates: int
    compiled_depth: int
    compiled_cx_count: int
    compiled_swap_count: int
    compiled_total_2q_gates: int
    compiled_gate_counts: dict[str, int]
    optimization_level: int
    basis_gates: list[str]
    seed_transpiler: int | None
    initial_layout: list[int] | None
    final_layout: list[int] | None
    transpilation_runtime_seconds: float
    circuit: QuantumCircuit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of metrics."""
        d = asdict(self)
        d.pop("circuit", None)
        return d


def validate_capacity(circuit: QuantumCircuit, processor_config: dict[str, Any]) -> None:
    """Verify that the logical circuit fits on the target hardware.
    
    Raises:
        CapacityError: If circuit.num_qubits > processor_config['num_qubits'].
    """
    n_logical = circuit.num_qubits
    n_physical = processor_config["num_qubits"]
    if n_logical > n_physical:
        raise CapacityError(
            f"Circuit requires {n_logical} logical qubits, but {processor_config['name']} "
            f"has only {n_physical} physical qubits. Cannot compile."
        )


def compile_for_processor(
    circuit: QuantumCircuit,
    processor_name_or_config: str | dict[str, Any],
    *,
    optimization_level: int = 1,
    seed_transpiler: int | None = 42,
    basis_gates: Sequence[str] | None = None,
    layout_method: str = "sabre",
    routing_method: str = "sabre",
) -> TranspilationResult:
    """Compile a logical circuit onto a target processor.
    
    Args:
        circuit: Logical QuantumCircuit (uncompiled).
        processor_name_or_config: 'processor_A', 'processor_B', or loaded config dict.
        optimization_level: Qiskit transpiler optimization level (default 1).
        seed_transpiler: Seed for reproducible layout and routing (default 42).
        basis_gates: Native gate set (default from processor config: ['rz', 'sx', 'x', 'cx']).
        layout_method: Initial layout strategy (default 'sabre').
        routing_method: Routing/SWAP insertion algorithm (default 'sabre').
        
    Returns:
        TranspilationResult with compiled circuit and resource metrics.
        
    Raises:
        CapacityError: If circuit width exceeds processor qubit count.
    """
    if isinstance(processor_name_or_config, str):
        config = load_processor(processor_name_or_config)
    else:
        config = processor_name_or_config

    validate_capacity(circuit, config)

    coupling_map = get_coupling_map(config)
    effective_basis = list(basis_gates) if basis_gates is not None else list(config.get("basis_gates", ["rz", "sx", "x", "cx"]))

    # Capture logical metrics before compilation
    logical_ops = circuit.count_ops()
    logical_2q = int(logical_ops.get("rzz", 0) + logical_ops.get("cx", 0) + logical_ops.get("cz", 0) + logical_ops.get("swap", 0))
    logical_depth = circuit.depth()

    t0 = time.perf_counter()
    compiled = transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=effective_basis,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        layout_method=layout_method,
        routing_method=routing_method,
    )
    runtime = time.perf_counter() - t0

    compiled_ops = {k: int(v) for k, v in compiled.count_ops().items()}
    compiled_depth = compiled.depth()
    compiled_cx = int(compiled_ops.get("cx", 0))
    compiled_swap = int(compiled_ops.get("swap", 0))

    # Total 2-qubit gates in the compiled basis
    compiled_2q = compiled_cx + compiled_swap

    # Layout extraction
    initial_layout = None
    final_layout = None
    if compiled.layout is not None:
        try:
            if hasattr(compiled.layout, "initial_layout") and compiled.layout.initial_layout is not None:
                initial_layout = [
                    compiled.find_bit(q).index
                    for q in compiled.layout.initial_layout.get_virtual_bits().keys()
                ]
            if hasattr(compiled.layout, "final_layout") and compiled.layout.final_layout is not None:
                final_layout = [
                    compiled.find_bit(q).index
                    for q in compiled.layout.final_layout.get_virtual_bits().keys()
                ]
        except Exception:
            pass

    return TranspilationResult(
        processor_name=config["name"],
        num_physical_qubits=config["num_qubits"],
        num_logical_qubits=circuit.num_qubits,
        logical_depth=logical_depth,
        logical_cx_or_2q_gates=logical_2q,
        compiled_depth=compiled_depth,
        compiled_cx_count=compiled_cx,
        compiled_swap_count=compiled_swap,
        compiled_total_2q_gates=compiled_2q,
        compiled_gate_counts=compiled_ops,
        optimization_level=optimization_level,
        basis_gates=effective_basis,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
        final_layout=final_layout,
        transpilation_runtime_seconds=runtime,
        circuit=compiled,
    )


def compare_processors(
    circuit: QuantumCircuit,
    processors: Sequence[str] = ("processor_A", "processor_B"),
    *,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> dict[str, Any]:
    """Compile the identical logical circuit onto multiple processors and compare metrics.
    
    If a processor cannot fit the circuit, the capacity failure is honestly reported
    rather than fabricating zero values or failing the entire study.
    """
    results: dict[str, Any] = {}
    for name in processors:
        config = load_processor(name)
        try:
            res = compile_for_processor(
                circuit,
                config,
                optimization_level=optimization_level,
                seed_transpiler=seed_transpiler,
            )
            results[name] = {
                "status": "SUCCESS",
                "metrics": res.to_dict(),
            }
        except CapacityError as exc:
            results[name] = {
                "status": "CAPACITY_EXCEEDED",
                "error": str(exc),
                "num_logical_qubits": circuit.num_qubits,
                "num_physical_qubits": config["num_qubits"],
            }
    return results
