import numpy as np
import torch
import pytest


def test_seeding_returns_rng():
    from qwarm.utils.seeding import set_global_seed
    rng = set_global_seed(0)
    assert isinstance(rng, np.random.Generator)


def test_seeding_reproducibility():
    from qwarm.utils.seeding import set_global_seed
    set_global_seed(42)
    a = np.random.default_rng(42).random()
    set_global_seed(42)
    b = np.random.default_rng(42).random()
    assert a == b


def test_torch_seeding():
    from qwarm.utils.seeding import set_global_seed
    set_global_seed(7)
    t1 = torch.rand(3).clone()
    set_global_seed(7)
    t2 = torch.rand(3).clone()
    assert torch.allclose(t1, t2)


def test_env_var_set():
    import os
    from qwarm.utils.seeding import set_global_seed
    set_global_seed(99)
    assert os.environ.get("QWARM_SEED") == "99"


def test_get_logger_returns_logger():
    import logging
    from qwarm.utils.logging import get_logger
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_idempotent():
    from qwarm.utils.logging import get_logger
    l1 = get_logger("same_name")
    l2 = get_logger("same_name")
    assert l1 is l2
