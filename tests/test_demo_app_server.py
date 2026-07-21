import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from demo_app.server import _rollout
from qwarm.agents.gnn_dqn import GNNDQN
from qwarm.env.dynamic_graph import DynamicGraph
from qwarm.utils.device import resolve_device


def test_rollout_flags_step_budget_exhaustion_not_genuine_stuck():
    """A tiny max_steps on a graph the agent can't solve that fast should set hit_step_budget."""
    device = resolve_device("cpu")
    g = DynamicGraph(grid_width=10, grid_height=10, extra_edges=2, seed=7)
    for i in range(1, 4):
        g.update_graph(iteration=i)
    agent = GNNDQN(node_in_dim=4, hidden_dim=32, device=device, seed=0)
    node_ids = list(g.nodes.keys())
    src, dst = node_ids[0], node_ids[-1]
    result = _rollout(agent, g, src, dst, max_steps=1)
    assert result["hit_step_budget"] is True
    assert result["reached"] is False
