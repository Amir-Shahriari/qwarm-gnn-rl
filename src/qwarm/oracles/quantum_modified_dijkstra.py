"""Quantum Modified Dijkstra oracle with graceful qiskit fallback (A1 design decision)."""
import heapq
import logging
import time
import numpy as np
from qwarm.oracles._cost import calculate_cost

try:
    from qiskit_aer import Aer as _Aer
    from qiskit import QuantumCircuit
    from qiskit import execute as _execute

    HAS_QISKIT: bool = True
except ImportError:
    HAS_QISKIT = False

_log = logging.getLogger(__name__)


def _numpy_phase_estimation(
    distance: float,
    travel_time: float,
    rng: np.random.Generator,
) -> float:
    """Hadamard + phase-kick simulation using numpy."""
    composite = distance + 0.1 * travel_time
    phases = np.array([composite / (2 ** i) for i in range(6)])
    # Hadamard superposition: start each qubit at 50/50
    bits_zero = (rng.random(6) < 0.5).astype(int)
    # Phase kick: flip with prob proportional to sin^2(phase/2)
    kick = (rng.random(6) < np.abs(np.sin(phases / 2)) ** 2).astype(int)
    encoded = np.clip(bits_zero + kick, 0, 1)
    quantum_metric = int("".join(str(b) for b in encoded), 2)
    return (quantum_metric / 512.0) * float(rng.uniform(0.9, 1.1))


class QuantumModifiedDijkstra:
    def __init__(self, nodes: dict, graph: dict, seed: int = 0) -> None:
        self.nodes = nodes
        self.graph = graph
        self._rng = np.random.default_rng(seed)
        if HAS_QISKIT:
            self._backend = _Aer.get_backend("qasm_simulator")
        else:
            _log.warning(
                "qiskit-aer not installed — QuantumModifiedDijkstra will use numpy "
                "phase-estimator fallback."
            )

    def _phase_estimation(self, distance: float, travel_time: float) -> float:
        if HAS_QISKIT:
            try:
                qc = QuantumCircuit(6, 6)
                qc.h(range(6))
                composite_metric = distance + 0.1 * travel_time
                for i in range(6):
                    qc.p(composite_metric / (2 ** i), i)
                qc.measure(range(6), range(6))
                result = _execute(qc, self._backend, shots=512).result()
                counts = result.get_counts()
                measured_str = max(counts, key=counts.get)
                return (int(measured_str, 2) / 512.0) * float(self._rng.uniform(0.9, 1.1))
            except Exception:
                pass
        return _numpy_phase_estimation(distance, travel_time, self._rng)

    def find_optimized_route(
        self,
        source: str,
        destination: str,
        diversity_scale: float = 1.0,
        congestion_scale: float = 1.0,
    ) -> tuple[float, list[str], float]:
        t0 = time.time()
        queue: list[tuple[float, str, list[str]]] = [(0.0, source, [source])]
        visited: set[str] = set()

        while queue:
            cost_so_far, current, path = heapq.heappop(queue)
            if current == destination:
                return cost_so_far, path, time.time() - t0
            if current in visited:
                continue
            visited.add(current)

            for neighbor, details in self.graph[current].items():
                if (
                    not details["active"]
                    or not self.nodes[neighbor]["active"]
                    or not self.nodes[current]["active"]
                ):
                    continue
                if neighbor in visited:
                    continue
                quantum_val = self._phase_estimation(details["distance"], details["time"])
                new_cost = calculate_cost(
                    distance=details["distance"],
                    travel_time=details["time"],
                    quantum_value=quantum_val,
                    previous_cost=cost_so_far,
                    path_length=len(path),
                    degree=len(self.graph[neighbor]),
                    node_penalty=self.nodes[neighbor]["node_penalty"],
                    diversity_scale=diversity_scale,
                    congestion_scale=congestion_scale,
                )
                heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

        return float("inf"), [], time.time() - t0
