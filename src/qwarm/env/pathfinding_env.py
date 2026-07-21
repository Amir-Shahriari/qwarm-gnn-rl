"""Pathfinding RL environment. Logic verbatim from legacy baseline."""
import heapq

import numpy as np


class PathfindingEnv:
    def __init__(
        self,
        graph: dict,
        nodes: dict,
        source: str,
        destination: str,
        max_steps: int = 200,
        lambda_shape: float = 0.0,
        realtime_perturb: bool = False,
        perturb_every_n_steps: int = 30,
        perturb_n_nodes: int = 2,
        invalid_penalty_mode: str = "legacy",
        invalid_scaled_floor: float = -50.0,
        invalid_nonterminal_penalty: float = -40.0,
    ) -> None:
        if invalid_penalty_mode not in ("legacy", "scaled", "nonterminal"):
            raise ValueError(
                f"invalid_penalty_mode must be 'legacy', 'scaled' or 'nonterminal', "
                f"got {invalid_penalty_mode!r}"
            )
        self.graph = graph
        self.nodes = nodes
        self.source = source
        self.destination = destination
        self.max_steps = max_steps
        self.lambda_shape = lambda_shape
        self.realtime_perturb = realtime_perturb
        self.perturb_every_n_steps = perturb_every_n_steps
        self.perturb_n_nodes = perturb_n_nodes
        self.invalid_penalty_mode = invalid_penalty_mode
        self.invalid_scaled_floor = invalid_scaled_floor
        self.invalid_nonterminal_penalty = invalid_nonterminal_penalty
        self.current_node: str | None = None
        self.steps_taken: int = 0
        self.visited_nodes: set[str] = set()
        # "scaled" needs the reverse-Dijkstra distance map even without shaping.
        # NOTE: the map is computed once at construction; under realtime_perturb
        # it can go stale, in which case the floor acts as the fallback penalty.
        self._static_dist_to_goal: dict[str, float] = (
            self._compute_dist_to_goal()
            if (lambda_shape > 0.0 or invalid_penalty_mode == "scaled")
            else {}
        )
        self._step_counter: int = 0
        self._perturb_counter: int = 0
        self.graph_changed: bool = False

    def _compute_dist_to_goal(self) -> dict[str, float]:
        """Reverse Dijkstra from destination: d[node] = min cost to reach destination."""
        rev: dict[str, list[tuple[str, float]]] = {}
        for u, neighbors in self.graph.items():
            if not self.nodes[u]["active"]:
                continue
            for v, edata in neighbors.items():
                if edata["active"] and self.nodes[v]["active"]:
                    cost = edata["distance"] + 0.1 * edata["time"] + self.nodes[v]["node_penalty"]
                    rev.setdefault(v, []).append((u, cost))

        dist: dict[str, float] = {self.destination: 0.0}
        queue: list[tuple[float, str]] = [(0.0, self.destination)]
        visited: set[str] = set()
        while queue:
            d, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            for pred, edge_cost in rev.get(node, []):
                nd = d + edge_cost
                if nd < dist.get(pred, float("inf")):
                    dist[pred] = nd
                    heapq.heappush(queue, (nd, pred))
        return dist

    def reset(self) -> str:
        self.current_node = self.source
        self.steps_taken = 0
        self.visited_nodes = {self.source}
        self._step_counter = 0
        self._perturb_counter = 0
        self.graph_changed = False
        return self.current_node

    def _invalid_action_result(self) -> tuple[str, float, bool]:
        """Outcome of an invalid action (closed edge / inactive node / revisit).

        legacy:      -5, terminal (original behavior; default).
        scaled:      -(remaining optimal cost from current node), floored at
                     invalid_scaled_floor, terminal. Unreachable goal -> floor.
        nonterminal: invalid_nonterminal_penalty, agent stays put, episode
                     continues; the attempt consumes one step of the budget so
                     max_steps still caps repeated invalid actions.
        """
        if self.invalid_penalty_mode == "scaled":
            remaining = self._static_dist_to_goal.get(self.current_node, float("inf"))
            penalty = max(-remaining, self.invalid_scaled_floor)
            return self.current_node, penalty, True
        if self.invalid_penalty_mode == "nonterminal":
            self.steps_taken += 1
            done = self.steps_taken >= self.max_steps
            return self.current_node, self.invalid_nonterminal_penalty, done
        return self.current_node, -5.0, True

    def step(self, action: str) -> tuple[str, float, bool]:
        assert self.current_node is not None, "Call reset() before step()"
        if action not in self.graph[self.current_node]:
            return self._invalid_action_result()

        edge_data = self.graph[self.current_node][action]
        if not (edge_data["active"] and self.nodes[action]["active"]):
            return self._invalid_action_result()

        if action in self.visited_nodes:
            return self._invalid_action_result()

        dist = edge_data["distance"]
        t_val = edge_data["time"]
        node_penalty = self.nodes[action]["node_penalty"]
        step_cost = dist + 0.1 * t_val + node_penalty
        reward = -step_cost

        if self.lambda_shape > 0.0:
            d_prev = self._static_dist_to_goal.get(self.current_node, float("inf"))
            d_next = self._static_dist_to_goal.get(action, float("inf"))
            if d_prev != float("inf") and d_next != float("inf"):
                reward += self.lambda_shape * (d_prev - d_next)

        self.current_node = action
        self.steps_taken += 1
        self.visited_nodes.add(action)

        if self.current_node == self.destination:
            reward += 100.0
            return self.current_node, reward, True

        done = self.steps_taken >= self.max_steps
        self._step_counter += 1
        if (self.realtime_perturb and not done
                and self._step_counter > 0
                and self._step_counter % self.perturb_every_n_steps == 0):
            self._apply_realtime_perturbation()
            self.graph_changed = True
        if not self.nodes[self.current_node]["active"]:
            return self.current_node, -10.0, True
        return self.current_node, reward, done

    def get_valid_actions(self) -> list[str]:
        assert self.current_node is not None
        return [
            nb
            for nb, data in self.graph[self.current_node].items()
            if data["active"] and self.nodes[nb]["active"]
        ]

    def _apply_realtime_perturbation(self) -> None:
        active = [n for n, d in self.nodes.items() if d["active"]]
        candidates = [n for n in active
                      if n not in (self.source, self.destination, self.current_node)]
        if not candidates:
            return
        k = min(self.perturb_n_nodes, len(candidates))
        to_deact = np.random.choice(candidates, size=k, replace=False)
        for n in to_deact:
            self.nodes[n]["active"] = False
            for nbr in self.graph.get(n, {}):
                self.graph[n][nbr]["active"] = False
                if nbr in self.graph and n in self.graph[nbr]:
                    self.graph[nbr][n]["active"] = False
        self._perturb_counter += 1

    @property
    def perturb_count(self) -> int:
        return self._perturb_counter
