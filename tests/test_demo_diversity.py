"""Tests for demonstration-diversity metrics computed at buffer-seeding time."""
import pytest

from qwarm.replay.expert_replay_buffer import Transition
from qwarm.training.expert_seeding import compute_demo_diversity


def _entry(path, oracle):
    transitions = [
        Transition(
            state_node=path[i], action_node=path[i + 1], reward=-1.0,
            next_state_node=path[i + 1], done=True, valid_next_actions=[],
            is_expert=True, iteration_added=0, goal_node=path[-1],
        )
        for i in range(len(path) - 1)
    ]
    return {"cost": 1.0, "path": path, "oracle": oracle, "state": 0,
            "wall_ms": 0.0, "transitions": transitions}


def test_empty_library():
    m = compute_demo_diversity([])
    assert m["n_unique_paths"] == 0
    assert m["mean_pairwise_jaccard_distance"] is None
    assert m["n_unique_state_actions"] == 0
    assert m["n_quantum_paths"] == 0 and m["n_classical_paths"] == 0


def test_single_path_has_no_pairwise_distance():
    m = compute_demo_diversity([_entry(["A", "B", "C"], "ClassicalAStar")])
    assert m["n_unique_paths"] == 1
    assert m["mean_pairwise_jaccard_distance"] is None
    assert m["n_unique_state_actions"] == 2  # (A,B), (B,C)


def test_jaccard_identical_node_sets_is_zero():
    lib = [_entry(["A", "B", "C"], "ClassicalAStar"),
           _entry(["A", "C", "B"], "quantum_inspired_stochastic")]
    m = compute_demo_diversity(lib)
    assert m["mean_pairwise_jaccard_distance"] == pytest.approx(0.0)


def test_jaccard_disjoint_node_sets_is_one():
    lib = [_entry(["A", "B"], "ClassicalAStar"),
           _entry(["C", "D"], "ClassicalAStar")]
    m = compute_demo_diversity(lib)
    assert m["mean_pairwise_jaccard_distance"] == pytest.approx(1.0)


def test_jaccard_partial_overlap():
    # {A,B,C} vs {B,C,D}: |∩|=2, |∪|=4 → distance 0.5
    lib = [_entry(["A", "B", "C"], "ClassicalAStar"),
           _entry(["B", "C", "D"], "faithful_qaoa")]
    m = compute_demo_diversity(lib)
    assert m["mean_pairwise_jaccard_distance"] == pytest.approx(0.5)


def test_state_action_coverage_dedups_across_paths():
    lib = [_entry(["A", "B", "C"], "ClassicalAStar"),
           _entry(["A", "B", "D"], "KShortestPaths")]  # (A,B) shared
    m = compute_demo_diversity(lib)
    assert m["n_unique_state_actions"] == 3  # (A,B), (B,C), (B,D)
    assert m["n_transitions"] == 4


def test_source_classification():
    lib = [
        _entry(["A", "B"], "ClassicalAStar"),
        _entry(["A", "C"], "KShortestPaths"),
        _entry(["A", "D"], "quantum_inspired_stochastic"),
        _entry(["A", "E"], "faithful_qaoa"),
        _entry(["A", "F"], "faithful_qaoa"),
    ]
    m = compute_demo_diversity(lib)
    assert m["n_classical_paths"] == 2
    assert m["n_quantum_paths"] == 3
    assert m["n_qaoa_paths"] == 2
    assert m["paths_by_source"] == {
        "ClassicalAStar": 1, "KShortestPaths": 1,
        "quantum_inspired_stochastic": 1, "faithful_qaoa": 2,
    }
