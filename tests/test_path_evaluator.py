import pytest
from qwarm.eval.path_evaluator import evaluate_agent, evaluate_tabular_agent
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.pyg_adapter import dynamic_graph_to_pyg


def test_evaluate_agent_returns_required_keys(tiny_graph):
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    data = dynamic_graph_to_pyg(tiny_graph)
    result = evaluate_agent(agent, tiny_graph, "Node_1", "Node_25", data)
    assert "reached_goal" in result
    assert "cost" in result
    assert "path" in result


def test_evaluate_agent_cost_inf_when_no_goal(tiny_graph):
    """An untrained agent probably won't reach the goal — cost must be inf."""
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    data = dynamic_graph_to_pyg(tiny_graph)
    result = evaluate_agent(agent, tiny_graph, "Node_1", "Node_25", data, max_steps=5)
    if not result["reached_goal"]:
        assert result["cost"] == float("inf")


def test_evaluate_agent_goal_adjacent(tiny_graph):
    """Agent that always picks 'Node_2' should reach goal when source=Node_1, goal=Node_2."""
    from unittest.mock import patch
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    data = dynamic_graph_to_pyg(tiny_graph)
    with patch.object(agent, "choose_action", return_value="Node_2"):
        result = evaluate_agent(agent, tiny_graph, "Node_1", "Node_2", data)
    assert result["reached_goal"] is True
    assert result["cost"] < float("inf")


def test_evaluate_tabular_agent_returns_required_keys(tiny_graph):
    agent = QLearningAgent()
    result = evaluate_tabular_agent(agent, tiny_graph, "Node_1", "Node_25")
    assert "reached_goal" in result
    assert "cost" in result
    assert "path" in result


def test_path_starts_at_source(tiny_graph):
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, seed=0)
    data = dynamic_graph_to_pyg(tiny_graph)
    result = evaluate_agent(agent, tiny_graph, "Node_1", "Node_25", data)
    assert result["path"][0] == "Node_1"
