import pytest
from qwarm.env.pathfinding_env import PathfindingEnv


def test_reset_returns_source(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    state = env.reset()
    assert state == "Node_1"
    assert env.steps_taken == 0


def test_step_valid_action_moves_agent(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    valid = env.get_valid_actions()
    assert len(valid) > 0
    action = valid[0]
    next_state, reward, done = env.step(action)
    assert next_state == action
    assert reward < 0  # negative cost


def test_step_invalid_action_penalizes(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    _, reward, done = env.step("NonExistentNode_9999")
    assert reward == -5
    assert done is True


def test_step_inactive_edge_penalizes(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    valid = env.get_valid_actions()
    nb = valid[0]
    tiny_graph.graph["Node_1"][nb]["active"] = False
    _, reward, done = env.step(nb)
    assert reward == -5
    assert done is True


def test_step_revisit_penalizes(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    valid = env.get_valid_actions()
    action = valid[0]
    env.step(action)
    env.current_node = "Node_1"
    _, reward, done = env.step(action)
    assert reward == -5
    assert done is True


def test_goal_reached_gives_bonus(tiny_graph):
    # Put goal adjacent to start
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_2")
    env.reset()
    valid = env.get_valid_actions()
    assert "Node_2" in valid
    _, reward, done = env.step("Node_2")
    assert reward > 0  # +50 goal bonus minus small step cost
    assert done is True


def test_max_steps_triggers_done(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25", max_steps=1)
    env.reset()
    valid = env.get_valid_actions()
    action = next(a for a in valid if a != "Node_25")
    _, reward, done = env.step(action)
    assert done is True


def test_get_valid_actions_excludes_inactive(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    nb = list(tiny_graph.graph["Node_1"].keys())[0]
    tiny_graph.graph["Node_1"][nb]["active"] = False
    valid = env.get_valid_actions()
    assert nb not in valid
