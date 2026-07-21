"""Run one training cell in an isolated subprocess with a hard timeout.

Motivation: a sweep cell that hangs (native code does not respond to Python-
level interruption) or dies in a C extension (no traceback) must fail loudly
and let the sweep continue, instead of stalling silently. Process isolation is
bit-compatible here because every cell fully re-seeds itself on entry
(set_global_seed + per-cell RNG construction).

The child process inherits the parent's console handles, so per-cell progress
prints keep flowing into the sweep log.
"""
from __future__ import annotations

import importlib
import multiprocessing
import traceback


def _worker(q, module_name: str, fn_name: str, kwargs: dict) -> None:
    try:
        mod = importlib.import_module(module_name)
        q.put(("ok", getattr(mod, fn_name)(**kwargs)))
    except Exception as exc:
        q.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_isolated(module_name: str, fn_name: str, kwargs: dict,
                 timeout_s: float) -> tuple[str, object]:
    """Execute module_name.fn_name(**kwargs) in a spawn subprocess.

    Returns (status, payload):
      ("ok", result)        — function returned normally
      ("error", traceback)  — function raised (Python-level)
      ("timeout", None)     — hard timeout; child terminated
      ("crashed", exitcode) — child died without reporting (native crash /
                              external kill)
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, module_name, fn_name, kwargs))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(30)
        if p.is_alive():
            p.kill()
            p.join()
        return ("timeout", None)
    try:
        return q.get(True, 10)
    except Exception:
        return ("crashed", p.exitcode)
