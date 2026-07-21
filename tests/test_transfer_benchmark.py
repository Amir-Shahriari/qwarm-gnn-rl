import pytest
from qwarm.eval.transfer_benchmark import run_transfer_benchmark


def test_transfer_benchmark_output_structure():
    results = run_transfer_benchmark(
        train_grid_config={"grid_width": 5, "grid_height": 5, "extra_edges": 0, "seed": 0},
        test_grid_configs=[
            {"grid_width": 5, "grid_height": 5, "extra_edges": 0, "seed": 1},
        ],
        n_train_iterations=1,
        episodes_per_iteration=5,
        n_eval_episodes=2,
        seed=0,
    )
    assert "gnn_warm" in results
    assert "tabular_warm" in results
    for agent_key in ["gnn_warm", "tabular_warm"]:
        for grid_key, metrics in results[agent_key].items():
            assert "goal_reach_rate" in metrics
            assert 0.0 <= metrics["goal_reach_rate"] <= 1.0


def test_transfer_multiple_grids():
    results = run_transfer_benchmark(
        train_grid_config={"grid_width": 5, "grid_height": 5, "extra_edges": 0, "seed": 0},
        test_grid_configs=[
            {"grid_width": 5, "grid_height": 5, "extra_edges": 0, "seed": 1},
            {"grid_width": 6, "grid_height": 4, "extra_edges": 0, "seed": 2},
        ],
        n_train_iterations=1,
        episodes_per_iteration=5,
        n_eval_episodes=2,
        seed=0,
    )
    assert len(results["gnn_warm"]) == 2
    assert len(results["tabular_warm"]) == 2
