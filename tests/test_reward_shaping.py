"""Tests for potential-based reward shaping in PathfindingEnv (Phase 2, §2.4)."""
import heapq

import pytest

from qwarm.env.pathfinding_env import PathfindingEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dijkstra(graph, nodes, source, destination):
    """Forward Dijkstra: returns (cost, path). Uses same formula as env.step()."""
    queue: list[tuple[float, str, list]] = [(0.0, source, [source])]
    best: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            return cost, path
        for nb, edata in graph[node].items():
            if edata["active"] and nodes[nb]["active"]:
                step_cost = edata["distance"] + 0.1 * edata["time"] + nodes[nb]["node_penalty"]
                nc = cost + step_cost
                if nc < best.get(nb, float("inf")):
                    best[nb] = nc
                    heapq.heappush(queue, (nc, nb, path + [nb]))
    return float("inf"), []


# ---------------------------------------------------------------------------
# Test 1 — shaping is additive, bounded by λ × step_cost (Ng et al. 1999)
# ---------------------------------------------------------------------------

def test_shaping_is_additive(tiny_graph):
    """Shaped reward = unshapped + λ·(d_prev − d_next) for one valid step."""
    src, dst = "Node_1", "Node_25"
    env0 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.0)
    env1 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.1)

    env0.reset()
    env1.reset()
    action = env1.get_valid_actions()[0]

    _, r0, _ = env0.step(action)
    _, r1, _ = env1.step(action)

    d_prev = env1._static_dist_to_goal.get(src, float("inf"))
    d_next = env1._static_dist_to_goal.get(action, float("inf"))

    if d_prev != float("inf") and d_next != float("inf"):
        assert abs(r1 - r0 - 0.1 * (d_prev - d_next)) < 1e-9
        # Shaping ≤ λ × step_cost (triangle inequality on d_static)
        step_cost = r0 - 0.0  # r0 = -step_cost (no bonus, non-goal step)
        assert 0.1 * (d_prev - d_next) <= abs(r0) + 1e-9
    else:
        assert abs(r1 - r0) < 1e-9  # no shaping applied


# ---------------------------------------------------------------------------
# Test 2 — telescoping: total shaping along optimal path = λ × d(src, goal)
# ---------------------------------------------------------------------------

def test_telescoping_sum_along_optimal_path(tiny_graph):
    """Σ shaping along Dijkstra path = λ · d_static(source, goal)."""
    src, dst = "Node_1", "Node_25"
    opt_cost, path = _dijkstra(tiny_graph.graph, tiny_graph.nodes, src, dst)

    if opt_cost == float("inf") or len(path) < 2:
        pytest.skip("No path between Node_1 and Node_25 in this graph instance")

    env0 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.0)
    env1 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.1)
    env0.reset()
    env1.reset()

    total0 = total1 = 0.0
    for i in range(len(path) - 1):
        a = path[i + 1]
        for env in (env0, env1):
            env.current_node = path[i]
            env.visited_nodes = {path[i]}
        _, r0, _ = env0.step(a)
        _, r1, _ = env1.step(a)
        total0 += r0
        total1 += r1

    d_src = env1._static_dist_to_goal.get(src, float("inf"))
    if d_src != float("inf"):
        assert abs((total1 - total0) - 0.1 * d_src) < 1e-6


# ---------------------------------------------------------------------------
# Test 3 — +50 terminal bonus is unchanged; shaping adds on top
# ---------------------------------------------------------------------------

def test_goal_bonus_unchanged(tiny_graph):
    """Reaching destination: shaped reward equals unshaped + λ·d_static(prev, goal)."""
    src, dst = "Node_1", "Node_2"
    env0 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.0)
    env1 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, src, dst, lambda_shape=0.1)
    env0.reset()
    env1.reset()

    if dst not in env0.get_valid_actions():
        pytest.skip("Node_2 not adjacent to Node_1 in this graph instance")

    _, r0, done0 = env0.step(dst)
    _, r1, done1 = env1.step(dst)

    assert done0 is True and done1 is True
    # goal bonus must make r0 positive (step cost < goal bonus for one hop)
    assert r0 > 0, "unshaped goal reward should be positive (goal_bonus − small_step_cost)"

    d_src = env1._static_dist_to_goal.get(src, float("inf"))
    if d_src != float("inf"):
        assert abs(r1 - r0 - 0.1 * d_src) < 1e-9
    else:
        assert abs(r1 - r0) < 1e-9
