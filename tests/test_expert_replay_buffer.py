import numpy as np
import pytest
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer, Transition
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.oracles.classical_astar import ClassicalAStar


def _t(is_expert: bool = False, iteration: int = 0) -> Transition:
    return Transition(
        state_node="A", action_node="B", reward=-5.0,
        next_state_node="B", done=False, valid_next_actions=["C"],
        is_expert=is_expert, iteration_added=iteration,
    )


def test_add_online_grows_online_pool():
    buf = ExpertReplayBuffer()
    buf.add_online_transition(_t(is_expert=False))
    assert len(buf.online_pool) == 1
    assert len(buf.expert_pool) == 0


def test_expert_pool_separate_from_online():
    buf = ExpertReplayBuffer()
    buf.expert_pool.append(_t(is_expert=True))
    buf.add_online_transition(_t(is_expert=False))
    assert len(buf.expert_pool) == 1
    assert len(buf.online_pool) == 1
    assert len(buf) == 2


def test_sample_respects_expert_ratio():
    buf = ExpertReplayBuffer(expert_ratio=0.3, rng=np.random.default_rng(0))
    for _ in range(200):
        buf.expert_pool.append(_t(is_expert=True))
    for _ in range(200):
        buf.online_pool.append(_t(is_expert=False))
    batch = buf.sample(100)
    n_expert = sum(1 for t in batch if t.is_expert)
    assert 20 <= n_expert <= 40  # 30% ± 10


def test_sample_falls_back_to_online_when_expert_empty():
    buf = ExpertReplayBuffer(rng=np.random.default_rng(0))
    for _ in range(50):
        buf.online_pool.append(_t(is_expert=False))
    batch = buf.sample(20)
    assert len(batch) == 20
    assert all(not t.is_expert for t in batch)


def test_sample_falls_back_to_expert_when_online_empty():
    buf = ExpertReplayBuffer(rng=np.random.default_rng(0))
    for _ in range(50):
        buf.expert_pool.append(_t(is_expert=True))
    batch = buf.sample(20)
    assert len(batch) == 20
    assert all(t.is_expert for t in batch)


def test_sample_smaller_than_pool_works():
    buf = ExpertReplayBuffer(rng=np.random.default_rng(0))
    for _ in range(5):
        buf.expert_pool.append(_t(is_expert=True))
    batch = buf.sample(3)
    assert len(batch) == 3


def test_prune_stale_drops_old():
    buf = ExpertReplayBuffer(staleness_window=5)
    for i in range(10):
        buf.expert_pool.append(_t(is_expert=True, iteration=i))
    dropped = buf.prune_stale(current_iteration=9)
    # cutoff = 9-5=4; iterations 0-3 dropped (4 items)
    assert dropped == 4
    assert all(t.iteration_added >= 4 for t in buf.expert_pool)


def test_prune_stale_noop_when_window_none():
    buf = ExpertReplayBuffer(staleness_window=None)
    for i in range(5):
        buf.expert_pool.append(_t(iteration=i))
    dropped = buf.prune_stale(current_iteration=10)
    assert dropped == 0
    assert len(buf.expert_pool) == 5


def test_determinism():
    buf1 = ExpertReplayBuffer(expert_ratio=0.5, rng=np.random.default_rng(42))
    buf2 = ExpertReplayBuffer(expert_ratio=0.5, rng=np.random.default_rng(42))
    for i in range(50):
        buf1.expert_pool.append(_t(iteration=i))
        buf2.expert_pool.append(_t(iteration=i))
    b1 = buf1.sample(10)
    b2 = buf2.sample(10)
    assert [t.iteration_added for t in b1] == [t.iteration_added for t in b2]


def test_add_expert_path_skips_invalid_transitions():
    """Transitions that return reward=-5 must not be stored."""
    g = DynamicGraph(5, 5, extra_edges=0, seed=0)
    # block all edges from Node_1
    for nb in list(g.graph["Node_1"].keys()):
        g.graph["Node_1"][nb]["active"] = False
        g.graph[nb]["Node_1"]["active"] = False
    buf = ExpertReplayBuffer()
    # Force a fake path that starts with an invalid step
    added = buf.add_expert_path(g, PathfindingEnv, ["Node_1", "Node_2"], iteration=0)
    assert added == 0


def test_add_expert_path_valid_path(tiny_graph):
    oracle = ClassicalAStar(tiny_graph.nodes, tiny_graph.graph)
    cost, path, _ = oracle.find_optimized_route("Node_1", "Node_25")
    if cost == float("inf"):
        pytest.skip("No path available in this graph state")
    buf = ExpertReplayBuffer()
    added = buf.add_expert_path(tiny_graph, PathfindingEnv, path, iteration=0)
    assert added > 0
    assert all(t.is_expert for t in buf.expert_pool)
