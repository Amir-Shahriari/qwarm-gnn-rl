"""Full pipeline smoke test: oracle → buffer → training → eval.

Must complete in < 5 minutes on CPU. Marked slow so it's excluded from the
fast CI tier but included in the smoke target.
"""
import pytest
import numpy as np
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer
from qwarm.training.train_gnn_dqn import train_gnn_dqn
from qwarm.eval.path_evaluator import evaluate_agent, evaluate_tabular_agent
from qwarm.eval.metrics import composite_route_cost, goal_reach_rate
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.training.train_tabular import train_qlearning_agent


@pytest.mark.slow
def test_full_pipeline_smoke():
    """End-to-end: oracle → expert replay → GNN training → honest evaluation."""
    g = DynamicGraph(5, 5, extra_edges=1, seed=0)
    oracle = ClassicalAStar(g.nodes, g.graph)
    qaoa = QuantumInspiredStochasticOracle(g.nodes, g.graph)

    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    buf = ExpertReplayBuffer(expert_ratio=0.5, rng=np.random.default_rng(0))

    logs = train_gnn_dqn(
        g, PathfindingEnv, agent, buf,
        oracles=[oracle, qaoa],
        queries=[("Node_1", "Node_25")],
        n_iterations=2, episodes_per_iteration=20,
        grad_steps_per_episode=2, batch_size=16, seed=0,
    )

    # Log structure checks
    assert "mean_return" in logs
    assert len(logs["mean_return"]) == 2
    assert logs["expert_pool_size"][-1] > 0

    # Honest evaluation
    data = dynamic_graph_to_pyg(g)
    result = evaluate_agent(agent, g, "Node_1", "Node_25", data)
    assert "reached_goal" in result
    assert "cost" in result
    assert result["path"][0] == "Node_1"
    if not result["reached_goal"]:
        assert result["cost"] == float("inf")

    # Metrics work
    if result["reached_goal"]:
        cost = composite_route_cost(result["path"], g)
        assert cost > 0.0

    print(f"  Smoke test: reached_goal={result['reached_goal']} cost={result['cost']:.2f}")


@pytest.mark.slow
def test_tabular_pipeline_smoke():
    """Tabular pipeline: train → honest evaluation."""
    g = DynamicGraph(5, 5, seed=0)
    agent = train_qlearning_agent(g, "Node_1", "Node_25", episodes=200)
    result = evaluate_tabular_agent(agent, g, "Node_1", "Node_25")
    assert "reached_goal" in result
    assert result["path"][0] == "Node_1"
    if not result["reached_goal"]:
        assert result["cost"] == float("inf")


@pytest.mark.slow
def test_expert_replay_buffer_grows_during_training():
    """Expert pool must be non-empty after warm training."""
    g = DynamicGraph(5, 5, seed=1)
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=1)
    buf = ExpertReplayBuffer(expert_ratio=0.5, rng=np.random.default_rng(1))
    oracle = ClassicalAStar(g.nodes, g.graph)
    train_gnn_dqn(
        g, PathfindingEnv, agent, buf, [oracle], [("Node_1", "Node_25")],
        n_iterations=1, episodes_per_iteration=5,
        grad_steps_per_episode=1, batch_size=8, seed=1,
    )
    assert len(buf.expert_pool) > 0
