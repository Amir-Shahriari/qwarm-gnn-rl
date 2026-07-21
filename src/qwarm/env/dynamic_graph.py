"""Dynamic grid graph environment. Logic verbatim from legacy baseline; uses
np.random.Generator instead of the global random module for determinism."""
from __future__ import annotations
import copy
import numpy as np


class DynamicGraph:
    def __init__(
        self,
        grid_width: int = 25,
        grid_height: int = 25,
        extra_edges: int = 2,
        max_distance: float = 30.0,
        max_time: float = 15.0,
        deactivate_prob: float = 0.1,
        node_deactivate_prob: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.deactivate_prob = deactivate_prob
        self.node_deactivate_prob = node_deactivate_prob
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_distance = max_distance
        self.max_time = max_time

        self.nodes: dict[str, dict] = {}
        node_id = 1
        for r in range(grid_height):
            for c in range(grid_width):
                self.nodes[f"Node_{node_id}"] = {
                    "coords": (float(c), float(r)),
                    "active": True,
                    "node_penalty": 0.0,
                }
                node_id += 1

        self.graph: dict[str, dict] = {node: {} for node in self.nodes}
        self.num_nodes = len(self.nodes)
        self._init_grid_edges(extra_edges)

    def _init_grid_edges(self, extra_edges: int) -> None:
        node_list = list(self.nodes.keys())

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                idx = row * self.grid_width + col
                current = node_list[idx]
                if col < self.grid_width - 1:
                    self._add_edge(current, node_list[idx + 1])
                if row < self.grid_height - 1:
                    self._add_edge(current, node_list[idx + self.grid_width])

        for node in node_list:
            for _ in range(extra_edges):
                neighbor = str(self._rng.choice(node_list))
                if neighbor != node:
                    self._add_edge(node, neighbor)

    def _add_edge(self, node_a: str, node_b: str) -> None:
        if node_b not in self.graph[node_a]:
            dist = float(self._rng.uniform(1.0, self.max_distance))
            tval = float(self._rng.uniform(1.0, self.max_time))
            self.graph[node_a][node_b] = {"distance": dist, "time": tval, "active": True}
            self.graph[node_b][node_a] = {"distance": dist, "time": tval, "active": True}

    def update_graph(
        self,
        factor: float = 0.2,
        iteration: int = 1,
        peak_range: tuple[int, int] = (2, 3),
    ) -> None:
        for node, edges in self.graph.items():
            for neighbor, details in edges.items():
                if self._rng.random() < self.deactivate_prob:
                    details["active"] = False
                else:
                    details["active"] = True
                    details["distance"] *= float(self._rng.uniform(1 - factor, 1 + factor))
                    details["time"] *= float(self._rng.uniform(1 - factor, 1 + factor))
                    if peak_range[0] <= iteration <= peak_range[1]:
                        details["time"] *= 1.5

        for node, info in self.nodes.items():
            if self._rng.random() < self.node_deactivate_prob:
                info["active"] = False
            else:
                info["active"] = True
            info["node_penalty"] = float(self._rng.uniform(0.0, 5.0))

    def save_state(self) -> dict:
        """Return a deep copy of all mutable graph/node state including RNG."""
        return {
            "graph": copy.deepcopy(self.graph),
            "nodes": copy.deepcopy(self.nodes),
            "rng_state": self._rng.bit_generator.state,
        }

    def restore_state(self, state: dict) -> None:
        """Restore graph/node state from a previously saved snapshot.

        Mutates self.graph and self.nodes in-place so that any external objects
        holding references to these dicts (e.g. oracle objects) automatically
        see the restored state — avoids stale-reference bugs after discover_all_paths.
        """
        self.graph.clear()
        self.graph.update(copy.deepcopy(state["graph"]))
        self.nodes.clear()
        self.nodes.update(copy.deepcopy(state["nodes"]))
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]

    def is_edge_active(self, node: str, neighbor: str) -> bool:
        return (
            self.graph[node][neighbor]["active"]
            and self.nodes[node]["active"]
            and self.nodes[neighbor]["active"]
        )

    def get_node_penalty(self, node: str) -> float:
        return self.nodes[node]["node_penalty"] if self.nodes[node]["active"] else float("inf")
