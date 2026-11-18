"""Load tensor data with the restricted PyTorch deserializer."""

from __future__ import annotations

from os import PathLike

from packaging.version import Version


def load_tensor_file(path: str | PathLike, *, map_location="cpu", weights_only: bool = True):
    """Load a packaged tensor artifact using a patched PyTorch runtime."""
    import torch

    if Version(str(torch.__version__)) < Version("2.6.0"):
        raise RuntimeError("Use the patched PyTorch version in requirements.txt to load release artifacts")
    if weights_only is not True:
        raise ValueError("Release artifacts require weights-only loading")
    return torch.load(path, map_location=map_location, weights_only=True)
