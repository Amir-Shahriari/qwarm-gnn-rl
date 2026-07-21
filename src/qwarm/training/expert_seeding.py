"""Convert oracle paths to expert transitions and push them to the replay buffer.

Key design principle: transitions must be computed on the SAME graph state
where the oracle found the path. If we compute a path on graph state X and
then try to replay it on graph state Y (after more perturbations), edges that
were open in X may be closed in Y, causing env.step() to return reward=-50
and silently drop the transitions — leaving the warm buffer nearly empty.

The fix: save the graph state before perturbation, compute transitions
immediately on that state, then restore. Store the pre-computed Transition
objects so seed_buffer_from_path_library can add them directly.
"""
from __future__ import annotations

import networkx as nx

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition


def seed_buffer_from_oracles(
    dyn_graph: DynamicGraph,
    buffer: ExpertReplayBuffer,
    oracles: list,
    queries: list[tuple[str, str]],
    iteration: int,
) -> int:
    """Run every oracle on every query on the CURRENT graph state and push transitions.

    Returns the total number of transitions added.
    """
    total_added = 0
    for oracle in oracles:
        for src, dst in queries:
            try:
                cost, path, _ = oracle.find_optimized_route(src, dst)
            except Exception:
                continue
            if cost == float("inf") or len(path) < 2:
                continue
            added = buffer.add_expert_path(dyn_graph, PathfindingEnv, path, iteration)
            total_added += added
    return total_added


def _build_nx_graph(dyn_graph: DynamicGraph) -> nx.DiGraph:
    """Build a weighted networkx DiGraph from currently active edges only."""
    G: nx.DiGraph = nx.DiGraph()
    for node, info in dyn_graph.nodes.items():
        if info["active"]:
            G.add_node(node)
    for u, neighbours in dyn_graph.graph.items():
        for v, edata in neighbours.items():
            if (
                edata["active"]
                and dyn_graph.nodes[u]["active"]
                and dyn_graph.nodes[v]["active"]
            ):
                weight = (
                    edata["distance"]
                    + 0.1 * edata["time"]
                    + dyn_graph.nodes[v]["node_penalty"]
                )
                G.add_edge(u, v, weight=max(weight, 1e-6))
    return G


def _compute_transitions(
    dyn_graph: DynamicGraph,
    path: list[str],
    iteration: int,
    goal_node: str | None = None,
    gamma: float = 0.95,
) -> list[Transition]:
    """Compute MC-return transitions for an expert path on the CURRENT graph state.

    Each hop is stored with its full discounted return (done=True) so the agent
    directly learns Q(s_t, a_t) = MC_return without needing multi-step Bellman
    backup across N hops — the root cause of Q-value sign inversion on long paths.

    Args:
        dyn_graph: The DynamicGraph at the state where transitions are computed
        path: The oracle-computed path [src, ..., dst]
        iteration: Current iteration number
        goal_node: The goal/destination node (usually path[-1])
        gamma: Discount factor — must match the agent's gamma (default 0.95)

    Returns only the valid transitions (truncates at first inactive edge).
    """
    if len(path) < 2:
        return []

    if goal_node is None:
        goal_node = path[-1]

    env = PathfindingEnv(
        dyn_graph.graph, dyn_graph.nodes, path[0], path[-1], max_steps=9999, lambda_shape=0.0
    )
    env.reset()

    # Collect per-step rewards; stop at first inactive/invalid edge
    steps: list[tuple[str, str, float]] = []
    for i in range(len(path) - 1):
        s, a = path[i], path[i + 1]
        env.current_node = s
        env.visited_nodes = {s}
        _, reward, done = env.step(a)
        if reward == -5.0:  # inactive edge or revisit — truncate
            break
        steps.append((s, a, reward))
        if done:
            break

    if not steps:
        return []

    # Compute full discounted MC returns: G[t] = r[t] + γ·G[t+1]
    T = len(steps)
    mc_returns = [0.0] * T
    mc_returns[T - 1] = steps[T - 1][2]
    for t in range(T - 2, -1, -1):
        mc_returns[t] = steps[t][2] + gamma * mc_returns[t + 1]

    return [
        Transition(
            state_node=s,
            action_node=a,
            reward=G_t,
            next_state_node=a,   # unused — done=True skips bootstrapping
            done=True,
            valid_next_actions=[],
            is_expert=True,
            iteration_added=iteration,
            goal_node=goal_node,
        )
        for (s, a, _), G_t in zip(steps, mc_returns)
    ]


def discover_all_paths(
    dyn_graph: DynamicGraph,
    oracles: list,
    queries: list[tuple[str, str]],
    n_perturbation_states: int = 10,
    n_shortest_paths: int = 5,
    base_iteration: int = 0,
    gamma: float = 0.95,
) -> list[dict]:
    """Enumerate diverse paths across N perturbed graph states.

    For each snapshot:
    - Runs Yen's k-shortest-paths (networkx) for the top ``n_shortest_paths``
      structurally diverse routes.
    - Runs every oracle to get quantum-inspired alternatives.
    - Computes valid transitions IMMEDIATELY on that graph state so they
      remain valid regardless of future perturbations.
    - Restores the graph to its original state when done.

    Returns a list of dicts sorted by cost (ascending):
        {cost, path, oracle, state, wall_ms, transitions: list[Transition]}
    """
    seen: set[tuple[str, ...]] = set()
    all_entries: list[dict] = []

    initial_state = dyn_graph.save_state()

    for k in range(n_perturbation_states):
        dyn_graph.update_graph(iteration=base_iteration + k + 1)
        iter_tag = base_iteration + k + 1

        for src, dst in queries:
            # ── K-shortest paths (Yen's algorithm via networkx) ───────────
            # n_shortest_paths <= 0 disables Yen entirely (quantum_only arm:
            # no classical demonstrations may enter the buffer).
            G = _build_nx_graph(dyn_graph)
            if n_shortest_paths > 0 and G.has_node(src) and G.has_node(dst) and nx.has_path(G, src, dst):
                count = 0
                try:
                    for path in nx.shortest_simple_paths(G, src, dst, weight="weight"):
                        if count >= n_shortest_paths:
                            break
                        if path[-1] != dst:
                            continue
                        key = tuple(path)
                        if key not in seen:
                            seen.add(key)
                            nx_cost = sum(
                                G[path[i]][path[i + 1]]["weight"]
                                for i in range(len(path) - 1)
                            )
                            transitions = _compute_transitions(dyn_graph, path, iter_tag, goal_node=dst, gamma=gamma)
                            if transitions:
                                all_entries.append({
                                    "cost": nx_cost,
                                    "path": path,
                                    "oracle": "KShortestPaths",
                                    "state": k,
                                    "wall_ms": 0.0,
                                    "transitions": transitions,
                                })
                        count += 1
                except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                    pass

            # ── Oracle paths ──────────────────────────────────────────────
            for oracle in oracles:
                try:
                    import time as _time
                    t0 = _time.perf_counter()
                    cost, path, _ = oracle.find_optimized_route(src, dst)
                    wall_ms = (_time.perf_counter() - t0) * 1000
                except Exception:
                    continue
                if cost == float("inf") or len(path) < 2 or path[-1] != dst:
                    continue
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                transitions = _compute_transitions(dyn_graph, path, iter_tag, goal_node=dst, gamma=gamma)
                if transitions:
                    all_entries.append({
                        "cost": cost,
                        "path": path,
                        "oracle": getattr(oracle, "name", type(oracle).__name__),
                        "state": k,
                        "wall_ms": wall_ms,
                        "transitions": transitions,
                    })

    # Restore graph to its state before discovery
    dyn_graph.restore_state(initial_state)

    all_entries.sort(key=lambda d: d["cost"])
    return all_entries


def compute_demo_diversity(path_library: list[dict]) -> dict:
    """Demonstration-diversity metrics for a discover_all_paths() library.

    Returns:
        n_unique_paths: paths after dedup (discover already dedups by node tuple)
        mean_pairwise_jaccard_distance: 1 - |A∩B|/|A∪B| over node sets,
            averaged over all path pairs (None when < 2 paths)
        n_unique_state_actions: unique (state_node, action_node) pairs across
            all pre-computed transitions
        n_transitions: total transitions in the library
        paths_by_source: path count per provenance tag (oracle name / KShortestPaths)
        n_quantum_paths / n_classical_paths: grouped by QUANTUM_SOURCES /
            CLASSICAL_SOURCES (n_qaoa_paths counts faithful_qaoa alone)
    """
    from qwarm.oracles.pool import CLASSICAL_SOURCES, QUANTUM_SOURCES

    node_sets = [frozenset(entry["path"]) for entry in path_library]
    n = len(node_sets)

    jaccard_dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            union = len(node_sets[i] | node_sets[j])
            inter = len(node_sets[i] & node_sets[j])
            jaccard_dists.append(1.0 - (inter / union if union else 0.0))

    state_actions = {
        (t.state_node, t.action_node)
        for entry in path_library
        for t in entry.get("transitions", [])
    }

    by_source: dict[str, int] = {}
    for entry in path_library:
        src_tag = entry.get("oracle", "unknown")
        by_source[src_tag] = by_source.get(src_tag, 0) + 1

    return {
        "n_unique_paths": n,
        "mean_pairwise_jaccard_distance": (
            float(sum(jaccard_dists) / len(jaccard_dists)) if jaccard_dists else None
        ),
        "n_unique_state_actions": len(state_actions),
        "n_transitions": sum(len(e.get("transitions", [])) for e in path_library),
        "paths_by_source": by_source,
        "n_qaoa_paths": by_source.get("faithful_qaoa", 0),
        "n_quantum_paths": sum(by_source.get(s, 0) for s in QUANTUM_SOURCES),
        "n_classical_paths": sum(by_source.get(s, 0) for s in CLASSICAL_SOURCES),
    }


def seed_buffer_from_path_library(
    dyn_graph: DynamicGraph,
    buffer: ExpertReplayBuffer,
    path_library: list[dict],
    iteration: int,
) -> int:
    """Push every pre-computed expert transition into the buffer's expert pool.

    Uses the pre-computed Transition objects stored during discovery (which were
    computed on the correct graph state) rather than replaying paths on the
    current — potentially very different — graph state.
    """
    total = 0
    for entry in path_library:
        if "transitions" in entry and entry["transitions"]:
            for t in entry["transitions"]:
                buffer.expert_pool.append(t)
                total += 1
        else:
            # Fallback for entries without pre-computed transitions
            added = buffer.add_expert_path(
                dyn_graph, PathfindingEnv, entry["path"], iteration
            )
            total += added
    return total
