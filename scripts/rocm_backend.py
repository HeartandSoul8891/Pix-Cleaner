"""
AMD ROCm / HIP backend helpers for Pix-Cleaner.

Important:
PyTorch ROCm intentionally exposes AMD GPUs through the
torch.cuda API.

Therefore the device passed to Ultralytics is normally:

    cuda:0

even though the actual backend is ROCm/HIP.
"""

from __future__ import annotations

from typing import Any


def get_info() -> dict[str, Any]:
    """Return information about the first ROCm GPU."""

    try:
        import torch
    except ImportError:
        return {
            "backend": "ROCm",
            "available": False,
            "device": "cpu",
            "error": "PyTorch is not installed.",
        }

    hip_version = getattr(
        getattr(torch, "version", None),
        "hip",
        None,
    )

    if not hip_version:
        return {
            "backend": "ROCm",
            "available": False,
            "device": "cpu",
            "error": "This PyTorch installation is not a ROCm build.",
        }

    if not torch.cuda.is_available():
        return {
            "backend": "ROCm",
            "available": False,
            "device": "cpu",
            "hip_version": hip_version,
            "error": "ROCm/HIP is installed but no usable GPU is available.",
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
            "backend": "ROCm",
            "available": True,

            # This is deliberately cuda:0.
            "device": f"cuda:{device_index}",

            "gpu_name": getattr(
                properties,
                "name",
                "Unknown AMD GPU",
            ),

            "vram_gb": (
                float(total_memory) / (1024 ** 3)
                if total_memory
                else None
            ),

            "hip_version": hip_version,

            "torch_version": getattr(
                torch,
                "__version__",
                None,
            ),

            "architecture": getattr(
                properties,
                "gcnArchName",
                None,
            ),
        }

    except Exception as exc:
        return {
            "backend": "ROCm",
            "available": False,
            "device": "cpu",
            "hip_version": hip_version,
            "error": str(exc),
        }


def get_device() -> str:
    """Return the ROCm device string used by Ultralytics."""
    return get_info()["device"]


def get_recommended_batch_size(
    imgsz: int = 640,
    model_size: str = "medium",
) -> int:
    """
    Return a conservative initial ROCm batch-size recommendation.

    This is intentionally conservative.

    We can later replace this with a real VRAM-aware auto
    batch-size test once the basic backend is working.
    """

    info = get_info()

    if not info.get("available"):
        return 1

    vram = info.get("vram_gb")

    if not vram:
        return 1

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