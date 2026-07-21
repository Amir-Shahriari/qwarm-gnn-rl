"""Reproducible (source, goal) scenario sampler for multi-seed sweeps.

Generates scenario pairs with minimum Euclidean distance >= min_euclidean_fraction
of the grid diagonal so that every comparison is non-trivial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qwarm.env.dynamic_graph import DynamicGraph


@dataclass
class Scenario:
    grid_seed: int
    source_node: str
    destination_node: str
    euclidean_distance: float
    scenario_id: str


def _is_reachable(dyn_graph: DynamicGraph, source: str, destination: str) -> bool:
    """BFS reachability check on active subgraph."""
    if source == destination:
        return True
    visited: set[str] = {source}
    queue = [source]
    while queue:
        node = queue.pop()
        for nb, edge in dyn_graph.graph[node].items():
            if nb not in visited and edge["active"] and dyn_graph.nodes[nb]["active"]:
                if nb == destination:
                    return True
                visited.add(nb)
                queue.append(nb)
    return False


def sample_scenarios(
    grid_template: dict,
    n_scenarios: int,
    rng: np.random.Generator,
    min_euclidean_fraction: float = 0.6,
) -> list[Scenario]:
    """Sample n_scenarios (source, goal) pairs on a fresh DynamicGraph.

    Filters for:
    - Euclidean distance >= min_euclidean_fraction * grid_diagonal
    - Active-subgraph connectivity (BFS reachability)
    - source != destination

    Args:
        grid_template: dict with DynamicGraph constructor kwargs (grid_width,
            grid_height, extra_edges, deactivate_prob). seed is NOT included —
            the caller provides it via rng or grid_seed below.
        n_scenarios: number of valid scenario pairs to return.
        rng: reproducible random generator (should come from the outer seed).
        min_euclidean_fraction: source-to-goal Euclidean distance must be at
            least this fraction of the grid diagonal.

    Returns list of Scenario objects. Raises RuntimeError if n_scenarios valid
    pairs cannot be found within 5000 attempts.
    """
    grid_seed = int(rng.integers(0, 2**31))
    g = DynamicGraph(
        grid_width=grid_template["grid_width"],
        grid_height=grid_template["grid_height"],
        extra_edges=grid_template.get("extra_edges", 2),
        deactivate_prob=grid_template.get("deactivate_prob", 0.15),
        seed=grid_seed,
    )

    w = g.grid_width
    h = g.grid_height
    diagonal = math.sqrt((w - 1) ** 2 + (h - 1) ** 2)
    min_dist = min_euclidean_fraction * diagonal

    node_ids = list(g.nodes.keys())
    scenarios: list[Scenario] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = 5000

    while len(scenarios) < n_scenarios and attempts < max_attempts:
        attempts += 1
        src = str(rng.choice(node_ids))
        dst = str(rng.choice(node_ids))
        if src == dst or (src, dst) in seen:
            continue

        cx, cy = g.nodes[src]["coords"]
        dx, dy = g.nodes[dst]["coords"]
        dist = math.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
        if dist < min_dist:
            continue

        if not g.nodes[src]["active"] or not g.nodes[dst]["active"]:
            continue

        if not _is_reachable(g, src, dst):
            continue

        seen.add((src, dst))
        sid = f"seed{grid_seed}_s{len(scenarios)}"
        scenarios.append(Scenario(
            grid_seed=grid_seed,
            source_node=src,
            destination_node=dst,
            euclidean_distance=dist,
            scenario_id=sid,
        ))

    if len(scenarios) < n_scenarios:
        raise RuntimeError(
            f"Only found {len(scenarios)}/{n_scenarios} valid scenarios "
            f"after {max_attempts} attempts (min_dist={min_dist:.1f}, "
            f"diagonal={diagonal:.1f})"
        )

    return scenarios
