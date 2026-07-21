"""Quantum-inspired stochastic heuristic oracle (fast bulk path generator).

This oracle uses A*-style search with Gaussian noise injection at each node
expansion, emulating QAOA's non-deterministic exploration at low computational
cost. It is NOT a simulation of the QAOA variational quantum circuit — for a
faithful classical simulation of QAOA, see `faithful_qaoa.py`.

Generates bulk expert paths for the Expert Replay Buffer (typically 30+
paths per source-destination query across multiple graph snapshots).

Formerly named `NormalizedSimulatedQAOA`; renamed to accurately reflect its
classical, non-variational nature.
"""
import heapq
import random
import time
import numpy as np
from qwarm.oracles._cost import calculate_cost


class QuantumInspiredStochasticOracle:
    name = "quantum_inspired_stochastic"

    def __init__(self, nodes: dict, graph: dict) -> None:
        self.nodes = nodes
        self.graph = graph

    def heuristic(self, current: str, destination: str) -> float:
        cx, cy = self.nodes[current]["coords"]
        dx, dy = self.nodes[destination]["coords"]
        base_dist = float(np.hypot(cx - dx, cy - dy))
        congestion_penalty = len(self.graph[current]) * random.uniform(0.05, 0.1)
        quantum_variability = random.gauss(0, 0.05 * base_dist)
        return (
            (base_dist / 100.0 + congestion_penalty + quantum_variability)
            * random.uniform(0.9, 1.1)
        )

    def find_optimized_route(
        self,
        source: str,
        destination: str,
        diversity_scale: float = 1.0,
        congestion_scale: float = 1.0,
    ) -> tuple[float, list[str], float]:
        t0 = time.perf_counter()
        visited: set[str] = set()
        queue: list[tuple[float, str, list[str]]] = [(0.0, source, [source])]

        while queue:
            cost_so_far, current, path = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)

            if current == destination:
                return cost_so_far, path, time.perf_counter() - t0

            for neighbor, data in self.graph[current].items():
                if (
                    not data["active"]
                    or not self.nodes[neighbor]["active"]
                    or not self.nodes[current]["active"]
                ):
                    continue
                if neighbor in visited:
                    continue
                h = self.heuristic(neighbor, destination)
                for _ in range(3):
                    h *= random.uniform(0.99, 1.01)
                new_cost = calculate_cost(
                    distance=data["distance"],
                    travel_time=data["time"],
                    quantum_value=h,
                    previous_cost=cost_so_far,
                    path_length=len(path),
                    degree=len(self.graph[neighbor]),
                    node_penalty=self.nodes[neighbor]["node_penalty"],
                    diversity_scale=diversity_scale,
                    congestion_scale=congestion_scale,
                )
                heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

        return float("inf"), [], time.perf_counter() - t0
