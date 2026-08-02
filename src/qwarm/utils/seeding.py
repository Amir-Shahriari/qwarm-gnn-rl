import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed random, numpy, and torch; export QWARM_SEED for child processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Best-effort CUDA determinism — NOT a bit-reproducibility guarantee.
        # warn_only=True means ops with no deterministic implementation still
        # run, they only warn. Empirically two same-seed 25x25 sweeps
        # (runs/sweep_phase3_{final,traced}.json) diverge on 14/25 warm cells
        # by cost and 7/25 cold cells by goal-reach; don't claim otherwise.
        # WDDM note: CUDA graphs (deferred) would cut batch_infer overhead to
        # ~10-30 ms on Linux/WSL2; on Windows WDDM the driver adds ~15 µs per
        # kernel launch regardless — see GPU2 gate comment in test_batch_infer.py.
        # TODO(perf): CUDA graphs on Linux/WSL2 cuts batch_infer to 10-30 ms; deferred to PhD Y1
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["QWARM_SEED"] = str(seed)
    return np.random.default_rng(seed)
