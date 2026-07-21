"""Quantum A* oracle with graceful qiskit fallback (A1 design decision)."""
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


def _numpy_quantum_heuristic(
    current_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """QPE-style phase encoding simulation using numpy."""
    distance = float(np.hypot(
        current_coords[0] - dest_coords[0],
        current_coords[1] - dest_coords[1],
    ))
    phases = np.array([distance / (2 ** i) for i in range(6)])
    probs = np.abs(np.cos(phases / 2)) ** 2
    bits = (rng.random(6) < probs).astype(int)
    heuristic_int = int("".join(str(b) for b in bits), 2)
    return (heuristic_int / 512.0) * float(rng.uniform(0.9, 1.1))


class QuantumAStar:
    def __init__(self, nodes: dict, graph: dict, seed: int = 0) -> None:
        self.nodes = nodes
        self.graph = graph
        self._rng = np.random.default_rng(seed)
        if HAS_QISKIT:
            self._backend = _Aer.get_backend("qasm_simulator")
        else:
            _log.warning(
                "qiskit-aer not installed — QuantumAStar will use numpy phase-estimator fallback."
            )

    def _quantum_heuristic(self, current: str, destination: str) -> float:
        if HAS_QISKIT:
            try:
                qc = QuantumCircuit(6, 6)
                cx, cy = self.nodes[current]["coords"]
                dx, dy = self.nodes[destination]["coords"]
                distance = float(np.hypot(cx - dx, cy - dy))
                for i in range(6):
                    qc.p(distance / (2 ** i), i)
                qc.measure(range(6), range(6))
                result = _execute(qc, self._backend, shots=512).result()
                counts = result.get_counts()
                measured_str = max(counts, key=counts.get)
                heuristic_int = int(measured_str, 2)
                return (heuristic_int / 512.0) * float(self._rng.uniform(0.9, 1.1))
            except Exception:
                pass
        return _numpy_quantum_heuristic(
            self.nodes[current]["coords"],
            self.nodes[destination]["coords"],
            self._rng,
        )

    def find_optimized_route(
        self,
        source: str,
        destination: str,
        diversity_scale: float = 1.0,
        congestion_scale: float = 1.0,
    ) -> tuple[float, list[str], float]:
        t0 = time.time()
        queue: list[tuple[float, str, list[str], float]] = [
            (0.0, source, [source], 0.0)
        ]
        visited: set[str] = set()

        while queue:
            f_cost, current, path, g_cost = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)

            if current == destination:
                return f_cost, path, time.time() - t0

            for neighbor, data in self.graph[current].items():
                if (
                    not data["active"]
                    or not self.nodes[neighbor]["active"]
                    or not self.nodes[current]["active"]
                ):
                    continue
                if neighbor in visited:
                    continue
                new_g = calculate_cost(
                    distance=data["distance"],
                    travel_time=data["time"],
                    quantum_value=0.0,
                    previous_cost=g_cost,
                    path_length=len(path),
                    degree=len(self.graph[neighbor]),
                    node_penalty=self.nodes[neighbor]["node_penalty"],
                    diversity_scale=diversity_scale,
                    congestion_scale=congestion_scale,
                )
                h = self._quantum_heuristic(neighbor, destination)
                heapq.heappush(
                    queue, (new_g + h, neighbor, path + [neighbor], new_g)
                )

        return float("inf"), [], time.time() - t0
