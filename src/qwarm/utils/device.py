"""Single source of truth for device selection and defensive device assertions."""
from __future__ import annotations

import torch


def resolve_device(preferred: str | torch.device = "auto") -> torch.device:
    """Return the best available device matching `preferred`.

    "auto" → cuda:0 → mps → cpu, in that order.
    """
    if isinstance(preferred, torch.device):
        return preferred
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def assert_on(device: torch.device, *modules_or_tensors) -> None:
    """Raise RuntimeError if any argument is not on `device`. For hot-path debug checks."""
    for x in modules_or_tensors:
        if isinstance(x, torch.nn.Module):
            for p in x.parameters():
                if p.device.type != device.type:
                    raise RuntimeError(
                        f"Expected module on {device}, found param on {p.device}"
                    )
        elif isinstance(x, torch.Tensor):
            if x.device.type != device.type:
                raise RuntimeError(
                    f"Expected tensor on {device}, found it on {x.device}"
                )
