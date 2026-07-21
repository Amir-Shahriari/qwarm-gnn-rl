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
    args = parser.parse_args()

    _check_deps()
    _check_checkpoints()

    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    print(f"[run_demo] Starting server at {url} ...")
    uvicorn.run("demo_app.server:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
