import pytest
import random as stdlib_random
from qwarm.agents.tabular_q import QLearningAgent
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.training.train_tabular import run_episode, train_qlearning_agent


def test_q_value_starts_at_zero():
    agent = QLearningAgent()
    assert agent.get_q_value("A", "B") == 0.0


def test_q_table_updates_after_step(tiny_graph):
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    state = env.reset()
    valid = env.get_valid_actions()
    action = valid[0]
    next_s, reward, done = env.step(action)
    next_valid = env.get_valid_actions() if not done else []
    agent.update_q(state, action, reward, next_s, next_valid)
    assert agent.get_q_value(state, action) != 0.0


def test_epsilon_zero_always_greedy():
    agent = QLearningAgent(epsilon=0.0)
    agent.set_q_value("A", "X", 100.0)
    agent.set_q_value("A", "Y", 1.0)
    for _ in range(20):
        assert agent.choose_action("A", ["X", "Y"]) == "X"


def test_epsilon_one_always_random():
    stdlib_random.seed(0)
    agent = QLearningAgent(epsilon=1.0)
    agent.set_q_value("A", "X", 100.0)
    agent.set_q_value("A", "Y", 1.0)
    choices = {agent.choose_action("A", ["X", "Y"]) for _ in range(50)}
    assert len(choices) > 1


def test_run_episode_returns_reward_and_path(tiny_graph):
    agent = QLearningAgent()
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    reward, path = run_episode(env, agent)
    assert isinstance(reward, float)
    assert isinstance(path, list)
    assert path[0] == "Node_1"


def test_train_qlearning_agent_populates_table(tiny_graph):
    agent = train_qlearning_agent(tiny_graph, "Node_1", "Node_25", episodes=30)
    assert len(agent.q_table) > 0


def test_train_qlearning_accepts_existing_agent(tiny_graph):
    agent = QLearningAgent(alpha=0.1)
    result = train_qlearning_agent(tiny_graph, "Node_1", "Node_25", episodes=10, agent=agent)
    assert result is agent


def test_q_update_bellman(tiny_graph):
    agent = QLearningAgent(alpha=1.0, gamma=0.5, epsilon=0.0)
    agent.set_q_value("A", "B", 0.0)
    agent.set_q_value("B", "C", 10.0)
    agent.update_q("A", "B", reward=5.0, next_state="B", next_valid_actions=["C"])
    # new_q = 0 + 1.0 * (5 + 0.5*10 - 0) = 10.0
    assert agent.get_q_value("A", "B") == pytest.approx(10.0)
