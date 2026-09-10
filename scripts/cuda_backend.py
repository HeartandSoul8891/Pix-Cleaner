
"""
NVIDIA CUDA backend helpers for Pix-Cleaner.

This module currently provides hardware information and the
Ultralytics device configuration.

Actual YOLO inference remains in ultralytics_script.py.
"""

from __future__ import annotations

from typing import Any


def get_info() -> dict[str, Any]:
    """Return information about the first CUDA GPU."""

    try:
        import torch
    except ImportError:
        return {
            "backend": "CUDA",
            "available": False,
            "device": "cpu",
            "error": "PyTorch is not installed.",
        }

    if not torch.cuda.is_available():
        return {
            "backend": "CUDA",
            "available": False,
            "device": "cpu",
            "error": "CUDA is not available.",
        }

    try:
        device_index = 0
        properties = torch.cuda.get_device_properties(device_index)

        total_memory = getattr(
            properties,
            "total_memory",
            0,
        )

        return {
            "backend": "CUDA",
            "available": True,
            "device": f"cuda:{device_index}",
            "gpu_name": getattr(
                properties,
                "name",
                "Unknown NVIDIA GPU",
            ),
            "vram_gb": (
                float(total_memory) / (1024 ** 3)
                if total_memory
                else None
            ),
            "cuda_version": getattr(
                torch.version,
                "cuda",
                None,
            ),
            "torch_version": getattr(
                torch,
                "__version__",
                None,
            ),
        }

    except Exception as exc:
        return {
            "backend": "CUDA",
            "available": False,
            "device": "cpu",
            "error": str(exc),
        }


def get_device() -> str:
    """Return the CUDA device used by Ultralytics."""
    return get_info()["device"]


def get_recommended_batch_size(
    imgsz: int = 640,
    model_size: str = "medium",
) -> int:
    """
    Return a conservative initial CUDA batch-size recommendation.

    This is intentionally NOT the final auto-batching algorithm.
    Later we can replace this with actual VRAM-aware probing.
    """

    info = get_info()

    if not info.get("available"):
        return 1

    vram = info.get("vram_gb")

    if not vram:
        return 1

    # Conservative starting point.
    if imgsz >= 1536:
        return 2

    if imgsz >= 1024:
        return 4

    if imgsz >= 800:
        return 8

    if imgsz >= 640:
        if vram >= 16:
            return 16
        if vram >= 12:
            return 12
        if vram >= 8:
            return 8
        return 4

    if vram >= 16:
        return 24

    if vram >= 12:
        return 16

    if vram >= 8:
        return 8

    return 4