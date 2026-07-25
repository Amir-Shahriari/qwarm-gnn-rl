#!/usr/bin/env python3
"""One-command launcher for the ICDM 2026 demo.

Usage: uv run python scripts/run_demo.py [--no-browser] [--port 8765]

Checks that dependencies are importable and that demo checkpoints exist,
then starts the FastAPI app and opens the browser once it's ready.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent

# `uv run python scripts/run_demo.py` puts scripts/ on sys.path[0], not ROOT,
# so the `demo_app` package (which lives at repo root, unpackaged) would not
# otherwise be importable by uvicorn's "demo_app.server:app" string import.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_deps() -> None:
    missing = []
    for mod in ("torch", "fastapi", "uvicorn", "torch_geometric", "networkx"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[run_demo] Missing dependencies: {missing}", file=sys.stderr)
        print("[run_demo] Run: uv sync --extra dev", file=sys.stderr)
        sys.exit(1)


def _check_checkpoints() -> None:
    manifest_path = ROOT / "demo_agents" / "manifest.json"
    if not manifest_path.exists():
        print(f"[run_demo] FATAL: {manifest_path} not found.", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    missing = []
    for scale_entry in manifest.values():
        for scenario in scale_entry["scenarios"]:
            for key in ("warm", "cold"):
                ckpt = ROOT / "demo_agents" / scenario[key]
                if not ckpt.exists():
                    missing.append(str(ckpt))
    if missing:
        print(f"[run_demo] FATAL: missing checkpoints: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"[run_demo] All checkpoints present ({manifest_path}).")


def _run_smoke() -> int:
    """Boot the app in-process and exercise one instance in each mode.

    Returns 0 if every mode responds cleanly, 1 on the first failure. This is
    the same coverage as tests/test_demo_smoke.py, wired into the one-command
    entry point so an operator can validate the booth with
    `uv run python scripts/run_demo.py --smoke` before opening it. It boots the
    full app (running the real startup precompute), so it fails loudly on a bad
    checkpoint, a broken rollout, or any unhandled server error.
    """
    from fastapi.testclient import TestClient

    from demo_app.server import app

    print("[smoke] Booting app and running one instance in each mode...")
    try:
        with TestClient(app) as client:
            cells = client.get("/cells").json()
            if not cells:
                print("[smoke] FAIL: no cells loaded", file=sys.stderr)
                return 1

            # Reach mode — one instance per scale.
            by_scale = {}
            for c in cells:
                by_scale.setdefault(c["scale"], c)
            for scale, cell in by_scale.items():
                r = client.post("/reach", json={"cell_id": cell["cell_id"], "perturb_multiplier": 1.0})
                if r.status_code != 200 or "warm" not in r.json():
                    print(f"[smoke] FAIL: reach {scale} -> {r.status_code} {r.text[:200]}", file=sys.stderr)
                    return 1
            print(f"[smoke] reach OK ({', '.join(sorted(by_scale))})")

            # Oracle-source mode — one solvable 25x25 instance, all three arms.
            oracle_cell = next((c for c in cells if c["scale"] == "25x25" and c["oracle_available"]), None)
            if oracle_cell is None:
                print("[smoke] FAIL: no oracle-available 25x25 cell", file=sys.stderr)
                return 1
            for arm in ("full_pool", "classical_only", "quantum_only"):
                r = client.post("/oracle_reach", json={"cell_id": oracle_cell["cell_id"], "oracle_arm": arm})
                if r.status_code != 200 or "warm" not in r.json():
                    print(f"[smoke] FAIL: oracle {arm} -> {r.status_code} {r.text[:200]}", file=sys.stderr)
                    return 1
            print("[smoke] oracle-source OK (full_pool, classical_only, quantum_only)")

            # Sandbox mode — training pair on the first scenario.
            scenarios = client.get("/sandbox/scenarios").json()
            grid = client.get(f"/sandbox/grid?cell_id={scenarios[0]['cell_id']}").json()
            r = client.post("/sandbox", json={
                "blocked": [], "src_idx": grid["train_src_idx"],
                "dst_idx": grid["train_dst_idx"], "cell_id": scenarios[0]["cell_id"],
            })
            if r.status_code != 200 or "warm" not in r.json():
                print(f"[smoke] FAIL: sandbox -> {r.status_code} {r.text[:200]}", file=sys.stderr)
                return 1
            print("[smoke] sandbox OK")
    except Exception as e:  # noqa: BLE001 — smoke check reports any failure loudly
        print(f"[smoke] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("[smoke] ALL MODES PASSED")
    return 0


def _open_browser_when_ready(url: str, timeout_s: float = 30.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)
    print(f"[run_demo] Server did not become ready within {timeout_s}s; open {url} manually.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Boot the app, run one instance in each mode, and exit nonzero on any error.",
    )
    args = parser.parse_args()

    _check_deps()
    _check_checkpoints()

    if args.smoke:
        sys.exit(_run_smoke())

    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    print(f"[run_demo] Starting server at {url} ...")
    uvicorn.run("demo_app.server:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
