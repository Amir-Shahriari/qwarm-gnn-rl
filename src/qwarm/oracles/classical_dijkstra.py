"""Classical Dijkstra oracle — verbatim logic from legacy baseline."""
import heapq
import time
from qwarm.oracles._cost import calculate_cost


class ClassicalDijkstra:
    def __init__(self, nodes: dict, graph: dict) -> None:
        self.nodes = nodes
        self.graph = graph

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
                new_cost = calculate_cost(
                    distance=details["distance"],
                    travel_time=details["time"],
                    quantum_value=0.0,
                    previous_cost=cost_so_far,
                    path_length=len(path),
                    degree=len(self.graph[neighbor]),
                    node_penalty=self.nodes[neighbor]["node_penalty"],
                    diversity_scale=diversity_scale,
                    congestion_scale=congestion_scale,
                )
                heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

        return float("inf"), [], time.time() - t0
