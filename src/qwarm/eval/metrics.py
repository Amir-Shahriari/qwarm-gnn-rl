"""Six acceptance-gate metrics."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch_geometric.data as pyg_data

from qwarm.env.dynamic_graph import DynamicGraph


@dataclass
class ReachabilityVerdict:
    reached_goal_strict: bool
    reached_goal_reasonable: bool
    cost: float
    dijkstra_reference_cost: float
    cost_ratio: float
    k_threshold: float


def _dijkstra_cost(dyn_graph: DynamicGraph, source: str, destination: str) -> float:
    """Unconstrained Dijkstra using the same edge-cost formula as evaluate_agent."""
    queue: list[tuple[float, str]] = [(0.0, source)]
    best: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    while queue:
        cost, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            return cost
        for nb, edge in dyn_graph.graph[node].items():
            if not edge["active"] or not dyn_graph.nodes[nb]["active"]:
                continue
            step_cost = (
                edge["distance"]
                + 0.1 * edge["time"]
                + dyn_graph.nodes[nb]["node_penalty"]
            )
            new_cost = cost + step_cost
            if new_cost < best.get(nb, float("inf")):
                best[nb] = new_cost
                heapq.heappush(queue, (new_cost, nb))
    return float("inf")


def evaluate_with_reasonableness(
    dyn_graph: DynamicGraph,
    agent,
    source: str,
    destination: str,
    k_threshold: float = 3.0,
    max_steps: int = 300,
    data: "pyg_data.Data | None" = None,
    library=None,
    lambda_retr: float = 0.5,
) -> ReachabilityVerdict:
    """Evaluate agent and flag whether cost is within k_threshold of Dijkstra optimal.

    reached_goal_strict: agent's path ends at destination.
    reached_goal_reasonable: strict AND cost <= k_threshold * dijkstra_cost.
    cost_ratio: agent_cost / dijkstra_cost (lower is better; 1.0 = optimal).
    """
    from qwarm.eval.path_evaluator import evaluate_agent

    result = evaluate_agent(agent, dyn_graph, source, destination, data, max_steps,
                            library=library, lambda_retr=lambda_retr)
    dijkstra_cost = _dijkstra_cost(dyn_graph, source, destination)

    strict = result["reached_goal"]
    agent_cost = result["cost"]
    ratio = agent_cost / max(dijkstra_cost, 1e-9)
    reasonable = strict and (agent_cost <= k_threshold * max(dijkstra_cost, 1e-9))

    return ReachabilityVerdict(
        reached_goal_strict=strict,
        reached_goal_reasonable=reasonable,
        cost=agent_cost,
        dijkstra_reference_cost=dijkstra_cost,
        cost_ratio=ratio,
        k_threshold=k_threshold,
    )


def composite_route_cost(
    path: list[str],
    dyn_graph: DynamicGraph,
    time_weight: float = 0.5,
    dist_weight: float = 0.3,
    congestion_weight: float = 0.2,
) -> float:
    if len(path) < 2:
        return float("inf")
    total_time = total_dist = total_cong = 0.0
    for i in range(len(path) - 1):
        edge = dyn_graph.graph[path[i]].get(path[i + 1])
        if edge is None or not edge["active"]:
            return float("inf")
        total_time += edge["time"]
        total_dist += edge["distance"]
        total_cong += dyn_graph.nodes[path[i + 1]]["node_penalty"]
    return (
        time_weight * total_time
        + dist_weight * total_dist
        + congestion_weight * total_cong
    )


def goal_reach_rate(eval_results: list[dict]) -> float:
    if not eval_results:
        return 0.0
    return sum(1.0 for r in eval_results if r["reached_goal"]) / len(eval_results)
