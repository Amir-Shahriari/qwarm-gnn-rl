"""Headless smoke test: boots the demo app in-process and exercises every endpoint once."""
from __future__ import annotations

import pathlib
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def test_smoke_reach_both_scales():
    from demo_app.server import app

    with TestClient(app) as client:
        cells = client.get("/cells").json()
        by_scale = {}
        for c in cells:
            by_scale.setdefault(c["scale"], c)

        assert set(by_scale) == {"25x25", "50x50"}
        for scale, cell in by_scale.items():
            resp = client.post("/reach", json={"cell_id": cell["cell_id"], "perturb_multiplier": 1.0})
            assert resp.status_code == 200, resp.text
            wire = resp.json()
            for agent_key in ("warm", "cold"):
                assert "reached" in wire[agent_key]
                assert "mean_latency_ms" in wire[agent_key]
                assert wire[agent_key]["mean_latency_ms"] >= 0.0


def test_smoke_oracle_source_arms_25x25():
    from demo_app.server import app

    with TestClient(app) as client:
        cells = client.get("/cells").json()
        oracle_cell = next(c for c in cells if c["scale"] == "25x25" and c["oracle_available"])
        for arm in ("full_pool", "classical_only", "quantum_only"):
            resp = client.post(
                "/oracle_reach", json={"cell_id": oracle_cell["cell_id"], "oracle_arm": arm}
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "warm" in body and "cold" in body and "dijkstra" in body
            assert "reached" in body["warm"]


def test_smoke_sandbox_endpoint():
    from demo_app.server import app

    with TestClient(app) as client:
        scenarios = client.get("/sandbox/scenarios").json()
        assert len(scenarios) > 0
        cell_id = scenarios[0]["cell_id"]
        grid = client.get(f"/sandbox/grid?cell_id={cell_id}").json()
        src_idx = grid["train_src_idx"]
        dst_idx = grid["train_dst_idx"]
        resp = client.post(
            "/sandbox", json={"blocked": [], "src_idx": src_idx, "dst_idx": dst_idx, "cell_id": cell_id}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "warm" in body and "cold" in body and "dijkstra" in body
        assert "reached" in body["warm"] and "reached" in body["cold"]
