"""Mixed-batch GNN-DQN training loop with persistent expert replay.

Per iteration:
  1. dyn_graph.update_graph()         — perturb environment
  2. seed_buffer_from_oracles()       — re-inject expert paths (if warm)
  3. buffer.prune_stale()             — drop stale expert transitions
  4. agent.encode()                   — refresh node embeddings
  5. Roll out episodes, push online transitions to buffer
  6. grad_steps_per_episode gradient updates from buffer.sample()

The cold-start variant is identical but with oracles=[] and
re_seed_experts_each_iteration=False.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition
from qwarm.training.expert_seeding import (
    seed_buffer_from_oracles,
    discover_all_paths,
    seed_buffer_from_path_library,
    compute_demo_diversity,
)
from qwarm.utils.seeding import set_global_seed


def train_gnn_dqn(
    dyn_graph: DynamicGraph,
    env_class,
    agent: GNNDQN,
    buffer: ExpertReplayBuffer,
    oracles: list,
    queries: list[tuple[str, str]],
    n_iterations: int = 10,
    episodes_per_iteration: int = 200,
    grad_steps_per_episode: int = 4,
    batch_size: int = 64,
    epsilon_start: float = 0.40,
    epsilon_end: float = 0.05,
    re_seed_experts_each_iteration: bool = True,
    seed: int = 0,
    trace_dir: "pathlib.Path | str | None" = None,
    pre_seed_n_states: int = 10,
    pre_seed_k_paths: int = 5,
    gamma: float = 0.95,
    env_kwargs: "dict | None" = None,
) -> dict:
    """Train a GNNDQN agent with mixed expert+online batches.

    Returns a per-iteration log dict with keys:
        iteration, mean_return, goal_reach_rate,
        expert_pool_size, online_pool_size, mean_loss
    """
    set_global_seed(seed)

    episode_writer = None
    if trace_dir is not None:
        pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)
        agent.open_trace(pathlib.Path(trace_dir) / "training_trace.jsonl")
        # Per-episode trace: one JSON line per rollout with the raw (unsmoothed)
        # return and goal flag, so learning curves can be plotted post hoc.
        episode_writer = open(
            pathlib.Path(trace_dir) / "episodes.jsonl", "w", buffering=1
        )

    logs: dict[str, list] = {
        "iteration": [],
        "mean_return": [],
        "goal_reach_rate": [],
        "expert_pool_size": [],
        "online_pool_size": [],
        "mean_loss": [],
    }

    # Pre-populate expert pool before any gradient step so the 40% expert ratio
    # is actually met from iteration 0. discover_all_paths samples multiple graph
    # perturbation states + k-shortest paths, giving O(100+) transitions vs the
    # ~2/iteration produced by the inline oracle seeding alone.
    if oracles and re_seed_experts_each_iteration and pre_seed_n_states > 0:
        path_library = discover_all_paths(
            dyn_graph, oracles, queries,
            n_perturbation_states=pre_seed_n_states,
            n_shortest_paths=pre_seed_k_paths,
            base_iteration=0,
            gamma=gamma,
        )
        seed_buffer_from_path_library(dyn_graph, buffer, path_library, iteration=0)

        # Demonstration-diversity snapshot of the seeded library (file write
        # only — consumes no RNG, so traced runs stay bit-compatible).
        if trace_dir is not None:
            with open(pathlib.Path(trace_dir) / "seeding_diversity.json", "w") as fh:
                json.dump(compute_demo_diversity(path_library), fh, indent=2)

        # Goal-adjacent seeding: for every node that can reach the destination in
        # one step, add a terminal transition with reward = -step_cost + 50.
        # This explicitly teaches Q(adj, goal) ≈ +35 so the agent never walks
        # past an immediately-reachable destination.
        for src, dst in queries:
            for node_id, neighbors in dyn_graph.graph.items():
                if dst not in neighbors:
                    continue
                edata = neighbors[dst]
                if not (edata["active"] and dyn_graph.nodes[node_id]["active"] and dyn_graph.nodes[dst]["active"]):
                    continue
                step_cost = edata["distance"] + 0.1 * edata["time"] + dyn_graph.nodes[dst]["node_penalty"]
                buffer.expert_pool.append(
                    Transition(
                        state_node=node_id,
                        action_node=dst,
                        reward=-step_cost + 100.0,
                        next_state_node=dst,
                        done=True,
                        valid_next_actions=[],
                        is_expert=True,
                        iteration_added=0,
                        goal_node=dst,
                    )
                )

    eps_decay = (epsilon_start - epsilon_end) / max(n_iterations - 1, 1)
    epsilon = epsilon_start
    global_episode = 0

    for iteration in range(n_iterations):
        # ── CRITICAL FIX: Seed experts BEFORE perturbing the graph ───────────────
        # Save the pre-perturbation state so expert transitions are valid
        state_before_perturb = dyn_graph.save_state()
        
        if re_seed_experts_each_iteration and oracles:
            # Compute expert paths on the CURRENT (unperturbed) graph state
            for oracle in oracles:
                for src, dst in queries:
                    try:
                        cost, path, _ = oracle.find_optimized_route(src, dst)
                    except Exception:
                        continue
                    if cost == float("inf") or len(path) < 2:
                        continue
                    # Compute transitions while graph is still unperturbed
                    added = buffer.add_expert_path(
                        dyn_graph, env_class, path, iteration, goal_node=dst,
                        gamma=gamma,
                    )
            buffer.prune_stale(iteration)
        
        # NOW perturb for the episode rollout
        dyn_graph.update_graph(iteration=iteration + 1)
        data = dynamic_graph_to_pyg(dyn_graph, device=agent.device)
        agent.encode(data)

        ep_returns: list[float] = []
        ep_goals: list[float] = []
        ep_losses: list[float] = []

        for _ in range(episodes_per_iteration):
            for src, dst in queries:
                env = env_class(
                    dyn_graph.graph, dyn_graph.nodes, src, dst, max_steps=400,
                    **(env_kwargs or {}),
                )
                state = env.reset()
                done = False
                ep_return = 0.0
                termination = None

                while not done:
                    valid = env.get_valid_actions()
                    if not valid:
                        termination = "dead_end"
                        break
                    action = agent.choose_action(state, valid, dst, data, epsilon=epsilon)
                    next_state, reward, done = env.step(action)
                    if done:
                        # Classification only reads env state — consumes no RNG,
                        # so traced runs stay bit-compatible with untraced ones.
                        # Precedence: goal > step_cap > invalid (agent stayed
                        # put) > other (node deactivated under the agent).
                        if next_state == dst:
                            termination = "goal"
                        elif env.steps_taken >= env.max_steps:
                            termination = "step_cap"
                        elif next_state == state:
                            termination = "invalid"
                        else:
                            termination = "other"
                    if getattr(env, "graph_changed", False):
                        data = dynamic_graph_to_pyg(dyn_graph, device=agent.device)
                        agent.encode(data)
                        env.graph_changed = False
                    ep_return += reward
                    next_valid = env.get_valid_actions() if not done else []
                    buffer.add_online_transition(
                        Transition(
                            state_node=state,
                            action_node=action,
                            reward=reward,
                            next_state_node=next_state,
                            done=done,
                            valid_next_actions=next_valid,
                            is_expert=False,
                            iteration_added=iteration,
                            goal_node=dst,  # ← Track the goal for this transition
                        )
                    )
                    state = next_state

                ep_returns.append(ep_return)
                ep_goals.append(float(state == dst))

                if episode_writer is not None:
                    episode_writer.write(json.dumps({
                        "global_episode": global_episode,
                        "iteration": iteration,
                        "return": float(ep_return),
                        "reached_goal": bool(state == dst),
                        "epsilon": float(epsilon),
                        "source": src,
                        "destination": dst,
                        "steps": int(env.steps_taken),
                        "termination": termination,
                    }) + "\n")
                global_episode += 1

                if len(buffer) >= batch_size:
                    for _ in range(grad_steps_per_episode):
                        batch = buffer.sample(batch_size)
                        goal = batch[0].goal_node if batch and batch[0].goal_node else dst
                        loss = agent.learn_from_batch(
                            batch, data, goal_node=goal,
                            trace_iteration=iteration,
                            trace_expert_buf_size=len(buffer.expert_pool),
                            trace_online_buf_size=len(buffer.online_pool),
                        )
                        ep_losses.append(loss)

        epsilon = max(epsilon_end, epsilon - eps_decay)

        logs["iteration"].append(iteration)
        logs["mean_return"].append(float(np.mean(ep_returns)))
        logs["goal_reach_rate"].append(float(np.mean(ep_goals)))
        logs["expert_pool_size"].append(len(buffer.expert_pool))
        logs["online_pool_size"].append(len(buffer.online_pool))
        logs["mean_loss"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)

    if trace_dir is not None:
        agent.close_trace()
    if episode_writer is not None:
        episode_writer.close()

    return logs
