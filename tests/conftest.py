import pytest
from qwarm.utils.seeding import set_global_seed
from qwarm.env.dynamic_graph import DynamicGraph


@pytest.fixture
def tiny_graph():
    """5x5 grid, seeded deterministically."""
    set_global_seed(0)
    return DynamicGraph(grid_width=5, grid_height=5, extra_edges=1, seed=0)


@pytest.fixture
def medium_graph():
    set_global_seed(1)
    return DynamicGraph(grid_width=25, grid_height=25, extra_edges=2, seed=1)
