"""Suite B — Zero-shot transfer benchmark.

Trains GNN warm and tabular warm on a 50×50 training grid, then evaluates
both agents without retraining on three unseen test grids. The expected
result is tabular_warm collapses to ~0 goal-reach (its Q-table keys don't
transfer) while GNN warm generalises via message-passing structure.
"""
from __future__ import annotations

import numpy as np

from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.eval.path_evaluator import evaluate_agent, evaluate_tabular_agent
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.training.train_tabular import train_qlearning_agent


def run_transfer_benchmark(
    train_grid_config: dict,
    test_grid_configs: list[dict],
    n_train_iterations: int = 10,
    episodes_per_iteration: int = 200,
    seed: int = 0,
    n_eval_episodes: int = 10,
    expert_ratio: float = 0.40,
) -> dict:
    """Train on train_grid, evaluate on each test_grid without retraining.

    Returns:
        {
          "gnn_warm":     {grid_key: {"goal_reach_rate": float}},
          "tabular_warm": {grid_key: {"goal_reach_rate": float}},
        }
    """
    train_grid = DynamicGraph(**train_grid_config)
    src = "Node_1"
    dst = f"Node_{train_grid.num_nodes}"
    queries = [(src, dst)]

    oracles = [
        ClassicalAStar(train_grid.nodes, train_grid.graph),
        QuantumInspiredStochasticOracle(train_grid.nodes, train_grid.graph),
    ]

    # Train GNN warm agent
    gnn = GNNDQN(node_in_dim=4, hidden_dim=128, seed=seed)
    buf = ExpertReplayBuffer(expert_ratio=expert_ratio, rng=np.random.default_rng(seed))
    train_gnn_dqn(
        train_grid, PathfindingEnv, gnn, buf, oracles, queries,
        n_iterations=n_train_iterations,
        episodes_per_iteration=episodes_per_iteration,
        re_seed_experts_each_iteration=True,
        seed=seed,
    )

    # Train tabular warm agent
    tabular = QLearningAgent()
    train_qlearning_agent(
        train_grid, src, dst,
        episodes=n_train_iterations * episodes_per_iteration,
        agent=tabular,
    )

    results: dict = {"gnn_warm": {}, "tabular_warm": {}}

    for cfg in test_grid_configs:
        test_grid = DynamicGraph(**cfg)
        test_src = "Node_1"
        test_dst = f"Node_{test_grid.num_nodes}"
        grid_key = f"{cfg['grid_width']}x{cfg['grid_height']}_seed{cfg.get('seed', 0)}"

        gnn_goals: list[float] = []
        tabular_goals: list[float] = []

        for _ in range(n_eval_episodes):
            test_grid.update_graph(iteration=1)
            test_data = dynamic_graph_to_pyg(test_grid)
            gr = evaluate_agent(gnn, test_grid, test_src, test_dst, test_data)
            gnn_goals.append(float(gr["reached_goal"]))
            tr = evaluate_tabular_agent(tabular, test_grid, test_src, test_dst)
            tabular_goals.append(float(tr["reached_goal"]))

        results["gnn_warm"][grid_key] = {"goal_reach_rate": float(np.mean(gnn_goals))}
        results["tabular_warm"][grid_key] = {"goal_reach_rate": float(np.mean(tabular_goals))}

    return results
