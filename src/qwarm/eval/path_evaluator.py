"""Honest path evaluator — checks path[-1] == destination; non-reaching paths cost inf.

Fixes the legacy get_path_from_agent() bug where timeout paths were silently
treated as valid.
"""
from __future__ import annotations

import torch_geometric.data as pyg_data

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg


def evaluate_agent(
    agent,
    dyn_graph: DynamicGraph,
    source: str,
    destination: str,
    data: pyg_data.Data | None = None,
    max_steps: int | None = None,
    library=None,
    lambda_retr: float = 0.5,
) -> dict:
    """Greedy rollout for a GNNDQN agent.

    When library is provided and the agent has select_action_with_library,
    blends network Q-values with retrieval Q-values (V2 warm path only).
    Cold agent always passes library=None so its path is byte-identical to V1.

    Returns:
        reached_goal: bool
        cost: float  (inf if goal not reached)
        path: list[str]
    """
    if data is None:
        data = dynamic_graph_to_pyg(dyn_graph)
    # Scale step budget to graph size so large grids don't time out prematurely
    if max_steps is None:
        max_steps = max(500, dyn_graph.num_nodes * 2)
    agent.encode(data)

    use_library = library is not None and hasattr(agent, "select_action_with_library")

    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0

    for _ in range(max_steps):
        valid = [
            nb
            for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        if use_library:
            action = agent.select_action_with_library(
                current, valid, destination, data, library, lambda_retr
            )
        else:
            action = agent.choose_action(current, valid, destination, data, epsilon=0.0)
        edge = dyn_graph.graph[current][action]
        cost += (
            edge["distance"]
            + 0.1 * edge["time"]
            + dyn_graph.nodes[action]["node_penalty"]
        )
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached_goal = path[-1] == destination
    return {
        "reached_goal": reached_goal,
        "cost": cost if reached_goal else float("inf"),
        "path": path,
    }


def evaluate_tabular_agent(
    agent,
    dyn_graph: DynamicGraph,
    source: str,
    destination: str,
    max_steps: int = 300,
) -> dict:
    """Greedy rollout for a QLearningAgent — same honest goal-check."""
    path = [source]
    visited: set[str] = {source}
    current = source
    cost = 0.0

    for _ in range(max_steps):
        valid = [
            nb
            for nb, d in dyn_graph.graph[current].items()
            if d["active"] and dyn_graph.nodes[nb]["active"] and nb not in visited
        ]
        if not valid:
            break
        q_vals = [(agent.get_q_value(current, a), a) for a in valid]
        action = max(q_vals, key=lambda x: x[0])[1]
        edge = dyn_graph.graph[current][action]
        cost += (
            edge["distance"]
            + 0.1 * edge["time"]
            + dyn_graph.nodes[action]["node_penalty"]
        )
        path.append(action)
        visited.add(action)
        current = action
        if current == destination:
            break

    reached_goal = path[-1] == destination
    return {
        "reached_goal": reached_goal,
        "cost": cost if reached_goal else float("inf"),
        "path": path,
    }
