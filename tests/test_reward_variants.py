"""Tests for config-flagged invalid-action reward variants (default OFF)."""
import pytest
from qwarm.env.pathfinding_env import PathfindingEnv


# ── Default (legacy) behavior unchanged ───────────────────────────────────────

def test_default_mode_is_legacy(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    assert env.invalid_penalty_mode == "legacy"
    env.reset()
    _, reward, done = env.step("NonExistentNode_9999")
    assert reward == -5.0
    assert done is True


def test_legacy_does_not_compute_dist_map(tiny_graph):
    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    assert env._static_dist_to_goal == {}


def test_unknown_mode_rejected(tiny_graph):
    with pytest.raises(ValueError):
        PathfindingEnv(
            tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
            invalid_penalty_mode="bogus",
        )


# ── scaled: penalty = -(remaining optimal cost), floored at -50, terminal ─────

def test_scaled_penalty_equals_remaining_cost(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="scaled",
    )
    env.reset()
    remaining = env._static_dist_to_goal["Node_1"]
    assert remaining < 50.0  # tiny 5x5 graph: remaining cost is small
    _, reward, done = env.step("NonExistentNode_9999")
    assert reward == pytest.approx(-remaining)
    assert done is True


def test_scaled_penalty_floored_at_minus_50(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="scaled",
    )
    env.reset()
    env._static_dist_to_goal["Node_1"] = 400.0  # simulate distant goal
    _, reward, done = env.step("NonExistentNode_9999")
    assert reward == -50.0
    assert done is True


def test_scaled_unreachable_goal_uses_floor(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="scaled",
    )
    env.reset()
    env._static_dist_to_goal.pop("Node_1", None)  # node not in dist map
    _, reward, done = env.step("NonExistentNode_9999")
    assert reward == -50.0
    assert done is True


def test_scaled_revisit_also_scaled(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="scaled",
    )
    env.reset()
    action = env.get_valid_actions()[0]
    env.step(action)
    env.current_node = "Node_1"
    _, reward, done = env.step(action)  # revisit
    assert reward <= 0.0
    assert reward >= -50.0
    assert done is True


# ── nonterminal: penalty -40, episode continues, step limit still caps ────────

def test_nonterminal_invalid_continues_episode(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="nonterminal",
    )
    env.reset()
    state, reward, done = env.step("NonExistentNode_9999")
    assert reward == -40.0
    assert done is False
    assert state == "Node_1"  # agent does not move
    assert env.steps_taken == 1  # invalid attempts consume the step budget


def test_nonterminal_valid_steps_unaffected(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="nonterminal",
    )
    env.reset()
    action = env.get_valid_actions()[0]
    next_state, reward, done = env.step(action)
    assert next_state == action
    assert -40.0 < reward < 0.0


def test_nonterminal_max_steps_caps_invalid_loop(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25",
        invalid_penalty_mode="nonterminal", max_steps=3,
    )
    env.reset()
    dones = []
    for _ in range(3):
        _, _, done = env.step("NonExistentNode_9999")
        dones.append(done)
    assert dones == [False, False, True]


def test_expert_path_truncation_mode_agnostic(tiny_graph):
    """add_expert_path must truncate at an inactive edge in every penalty mode,
    not just when the legacy -5 sentinel is returned."""
    import functools
    from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer

    env = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_25")
    env.reset()
    a = env.get_valid_actions()[0]
    env2 = PathfindingEnv(tiny_graph.graph, tiny_graph.nodes, a, "Node_25")
    env2.reset()
    b = next(x for x in env2.get_valid_actions() if x != "Node_1")
    path = ["Node_1", a, b]
    tiny_graph.graph[a][b]["active"] = False  # second hop is now invalid

    for mode in ("legacy", "scaled", "nonterminal"):
        buf = ExpertReplayBuffer()
        env_cls = functools.partial(PathfindingEnv, invalid_penalty_mode=mode)
        added = buf.add_expert_path(tiny_graph, env_cls, path, iteration=0)
        assert added == 1, f"mode={mode}: expected truncation after first hop"
        assert buf.expert_pool[0].action_node == a


def test_nonterminal_goal_bonus_unchanged(tiny_graph):
    env = PathfindingEnv(
        tiny_graph.graph, tiny_graph.nodes, "Node_1", "Node_2",
        invalid_penalty_mode="nonterminal",
    )
    env.reset()
    _, reward, done = env.step("Node_2")
    assert reward > 0.0  # +100 bonus minus step cost
    assert done is True
