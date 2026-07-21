"""Confirms the demo runs a full episode per scale with all outbound network blocked."""
from __future__ import annotations

import pathlib
import socket
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    return host in _LOOPBACK_HOSTS


@pytest.fixture
def block_network(monkeypatch):
    """Raise on any attempt to connect a real socket to a non-loopback address.

    Note: this patches connect()/connect_ex() rather than the socket.socket
    constructor itself. On Windows, asyncio's ProactorEventLoop creates a real
    loopback socketpair internally (via socket.socketpair's fallback) just to
    run TestClient's background portal thread -- that happens before any
    request is made and is unrelated to outbound network access, so blocking
    socket construction outright makes the test fail for a reason that has
    nothing to do with the app talking to the network. Blocking connect() to
    non-loopback addresses instead still catches any genuine outbound call.

    Known gap: on Windows, asyncio's ProactorEventLoop performs outbound
    connects via `_overlapped.ConnectEx` on the raw socket fd, bypassing
    `socket.socket.connect`/`connect_ex` entirely -- so this guard does not
    catch an async-native outbound call (e.g. `asyncio.open_connection`,
    or an `httpx.AsyncClient`/`aiohttp` request made through the default
    asyncio transport). demo_app/server.py has no such call today; if one is
    ever added, this test would need a loop-level guard too.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guarded_connect(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise OSError(
                f"Network access is blocked in this test (offline-operation check): {address!r}"
            )
        return real_connect(self, address, *args, **kwargs)

    def _guarded_connect_ex(self, address, *args, **kwargs):
        if not _is_loopback(address):
            raise OSError(
                f"Network access is blocked in this test (offline-operation check): {address!r}"
            )
        return real_connect_ex(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    yield
    monkeypatch.setattr(socket.socket, "connect", real_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", real_connect_ex)


def test_full_episode_warm_and_cold_both_scales_offline(block_network):
    from demo_app.server import app

    with TestClient(app) as client:
        cells = client.get("/cells").json()
        assert len(cells) > 0

        seen_scales = set()
        for cell in cells:
            if cell["scale"] in seen_scales:
                continue
            seen_scales.add(cell["scale"])
            resp = client.post("/reach", json={"cell_id": cell["cell_id"], "perturb_multiplier": 1.0})
            assert resp.status_code == 200
            wire = resp.json()
            assert "warm" in wire and "cold" in wire
            assert wire["warm"]["steps"] >= 0
            assert wire["cold"]["steps"] >= 0

        assert seen_scales == {"25x25", "50x50"}
