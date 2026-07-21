"""torch.compile wrapper with platform fallback (triton unavailable on Windows)."""
from __future__ import annotations

import logging
import sys

import torch

log = logging.getLogger(__name__)


def maybe_compile(
    module: torch.nn.Module,
    mode: str = "reduce-overhead",
) -> torch.nn.Module:
    """Apply torch.compile when supported; return module unchanged otherwise.

    Falls back to eager on:
    - Windows (triton not available)
    - CPU-only machines (compile gains are GPU-driven)
    - Any compile failure (logged as warning)
    """
    if sys.platform == "win32":
        log.info("torch.compile disabled on Windows (triton unavailable); using eager")
        return module
    if not torch.cuda.is_available():
        return module
    try:
        compiled = torch.compile(module, mode=mode)
        log.info(f"torch.compile applied (mode={mode})")
        return compiled
    except Exception as exc:
        log.warning(f"torch.compile failed ({exc!r}); falling back to eager")
        return module
