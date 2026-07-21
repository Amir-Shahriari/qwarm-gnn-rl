"""Classical A* oracle — verbatim logic from legacy baseline."""
import heapq
import time
import numpy as np
from qwarm.oracles._cost import calculate_cost


class ClassicalAStar:
    def __init__(self, nodes: dict, graph: dict) -> None:
        self.nodes = nodes
        self.graph = graph

    def heuristic(self, current: str, destination: str) -> float:
        cx, cy = self.nodes[current]["coords"]
        dx, dy = self.nodes[destination]["coords"]
        return float(np.hypot(cx - dx, cy - dy))

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
                h = self.heuristic(neighbor, destination)
                heapq.heappush(
                    queue, (new_g + h, neighbor, path + [neighbor], new_g)
                )

        return float("inf"), [], time.time() - t0
