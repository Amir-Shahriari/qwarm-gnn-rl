"""Integration test: ExpertLibrary collects paths from both oracle types."""
import pytest
import numpy as np

from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.env.pathfinding_env import PathfindingEnv
from qwarm.oracles.classical_astar import ClassicalAStar
from qwarm.oracles.quantum_inspired_stochastic import QuantumInspiredStochasticOracle
from qwarm.experts.library import ExpertPath, ExpertLibrary, build_expert_library
from qwarm.replay.expert_replay_buffer import ExpertReplayBuffer


@pytest.fixture
def g5():
    return DynamicGraph(5, 5, extra_edges=1, seed=0)


# ── ExpertPath dataclass ─────────────────────────────────────────────────────

def test_expert_path_stores_oracle_source():
    ep = ExpertPath(
        nodes=("Node_1", "Node_2", "Node_5"),
        cost=12.3,
        snapshot_id=0,
        oracle_source="classical_astar",
    )
    assert ep.oracle_source == "classical_astar"
    assert ep.nodes == ("Node_1", "Node_2", "Node_5")


# ── ExpertLibrary ────────────────────────────────────────────────────────────

def test_expert_library_composition_report():
    lib = ExpertLibrary()
    lib.add(ExpertPath(("A", "B"), 1.0, 0, "oracle_x"))
    lib.add(ExpertPath(("A", "C"), 2.0, 0, "oracle_y"))
    lib.add(ExpertPath(("A", "B"), 1.0, 0, "oracle_x"))
    report = lib.composition_report()
    assert report == {"oracle_x": 2, "oracle_y": 1}


def test_expert_library_len_and_iter():
    lib = ExpertLibrary()
    for i in range(3):
        lib.add(ExpertPath(("A", "B"), float(i), i, "src"))
    assert len(lib) == 3
    assert list(lib)[1].snapshot_id == 1


# ── build_expert_library ─────────────────────────────────────────────────────

def test_build_expert_library_tags_oracle_name(g5):
    """Each path is tagged with the oracle's .name attribute."""
    oracles = [
        ClassicalAStar(g5.nodes, g5.graph),
        QuantumInspiredStochasticOracle(g5.nodes, g5.graph),
    ]
    lib = build_expert_library(g5, oracles, "Node_1", "Node_25")
    assert len(lib) >= 1
    sources = {p.oracle_source for p in lib}
    assert "classical_astar" in sources or "quantum_inspired_stochastic" in sources
    # Every path must be oracle-tagged
    for p in lib:
        assert p.oracle_source != ""


def test_build_expert_library_returns_valid_paths(g5):
    """All collected paths must start at source and end at destination."""
    oracles = [ClassicalAStar(g5.nodes, g5.graph)]
    lib = build_expert_library(g5, oracles, "Node_1", "Node_25")
    for ep in lib:
        assert ep.nodes[0] == "Node_1"
        assert ep.nodes[-1] == "Node_25"
        assert ep.cost < float("inf")


# ── seed_replay_buffer ───────────────────────────────────────────────────────

def test_seed_replay_buffer_adds_transitions(g5):
    """seed_replay_buffer decomposes paths into transitions (oracle_source discarded)."""
    oracles = [ClassicalAStar(g5.nodes, g5.graph)]
    lib = build_expert_library(g5, oracles, "Node_1", "Node_25")
    buf = ExpertReplayBuffer(expert_ratio=0.5, rng=np.random.default_rng(0))
    n_added = lib.seed_replay_buffer(buf, g5, PathfindingEnv, iteration=0)
    assert n_added > 0
    assert len(buf.expert_pool) == n_added
    # Transitions in the buffer have no oracle_source field (it's discarded)
    from qwarm.replay.expert_replay_buffer import Transition
    t = buf.expert_pool[0]
    assert isinstance(t, Transition)
    assert not hasattr(t, "oracle_source")  # oracle_source never enters Transition


# ── mixed-oracle integration (requires qiskit) ───────────────────────────────

def test_expert_library_contains_both_oracle_sources():
    """After build, library has tagged paths from both fast heuristic AND faithful QAOA."""
    pytest.importorskip("qiskit_aer")
    from qwarm.oracles.faithful_qaoa import FaithfulSimulatedQAOA

    g = DynamicGraph(3, 3, extra_edges=0, seed=42)
    oracles = [
        QuantumInspiredStochasticOracle(g.nodes, g.graph),
        FaithfulSimulatedQAOA(g.nodes, g.graph, p_layers=1, k_candidate_paths=3,
                               max_edges_in_subgraph=10, n_optimiser_restarts=1,
                               max_optimiser_iters=20, seed=0),
    ]
    lib = build_expert_library(g, oracles, "Node_1", "Node_9")
    composition = lib.composition_report()
    # Both oracles should contribute at least one path
    assert "quantum_inspired_stochastic" in composition, (
        f"Fast heuristic missing from library. Composition: {composition}"
    )
    assert "faithful_qaoa" in composition, (
        f"Faithful QAOA missing from library. Composition: {composition}"
    )
    assert composition["faithful_qaoa"] >= 1
