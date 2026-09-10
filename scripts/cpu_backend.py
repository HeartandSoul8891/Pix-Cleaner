"""
CPU backend helpers for Pix-Cleaner.

This is the fallback when CUDA/ROCm GPU inference is unavailable.
"""

from __future__ import annotations

from typing import Any


def get_info() -> dict[str, Any]:
    """Return CPU backend information."""

    try:
        import torch

        torch_version = getattr(
            torch,
            "__version__",
            None,
        )

    except ImportError:
        torch_version = None

    return {
        "backend": "CPU",
        "available": True,
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": None,
        "torch_version": torch_version,
        "cuda_version": None,
        "hip_version": None,
        "architecture": None,
    }


def get_device() -> str:
    """Return the CPU device string."""
    return "cpu"


def get_recommended_batch_size(
    imgsz: int = 640,
    model_size: str = "medium",
) -> int:
    """
    Return a conservative CPU batch size.

    CPU batching behaves differently from GPU batching because
    memory is usually not the primary bottleneck.
    """

    if imgsz >= 1536:
        return 1

    if imgsz >= 1024:
        return 2

    if imgsz >= 800:
        return 2

    return 4