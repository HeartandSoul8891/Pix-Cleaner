"""
Pix-Cleaner hardware/backend detection.

Detects the best available inference backend:

    NVIDIA CUDA
    AMD ROCm / HIP
    CPU fallback

Important:
PyTorch ROCm exposes AMD GPUs through torch.cuda.
Therefore, both NVIDIA CUDA and AMD ROCm use a
"cuda:X" device string when passed to Ultralytics.

This module only detects hardware and reports backend
information. It does not perform YOLO inference.
"""

from __future__ import annotations

from typing import Any


def _load_torch():
    """Import PyTorch safely.

    Returns:
        torch module if installed, otherwise None.
    """
    try:
        import torch

        return torch
    except ImportError:
        return None


def _base_result(
    *,
    backend: str,
    device: str,
    available: bool,
    torch_version: str | None,
    cuda_version: str | None,
    hip_version: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Create a normalized backend result."""
    return {
        "backend": backend,
        "device": device,
        "available": available,
        "gpu_name": None,
        "gpu_count": 0,
        "vram_gb": None,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "hip_version": hip_version,
        "compute_capability": None,
        "architecture": None,
        "reason": reason,
    }


def detect_backend() -> dict[str, Any]:
    """Detect the best available inference backend.

    Returns a normalized dictionary containing:

        backend
        device
        available

        gpu_name
        gpu_count
        vram_gb

        torch_version
        cuda_version
        hip_version

        compute_capability
        architecture

        reason

    Examples:

        NVIDIA RTX 4090:

            {
                "backend": "CUDA",
                "device": "cuda:0",
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_count": 1,
                "vram_gb": 23.98,
                "cuda_version": "13.2",
                "compute_capability": "8.9",
                "architecture": None,
                ...
            }

        AMD RX 9070 XT:

            {
                "backend": "ROCm",
                "device": "cuda:0",
                "gpu_name": "AMD Radeon RX 9070 XT",
                "gpu_count": 1,
                "vram_gb": ...,
                "hip_version": "7.14.60850",
                "compute_capability": None,
                "architecture": "gfx1201",
                ...
            }
    """

    torch = _load_torch()

    # ---------------------------------------------------------
    # PyTorch not installed
    # ---------------------------------------------------------

    if torch is None:
        return _base_result(
            backend="CPU",
            device="cpu",
            available=False,
            torch_version=None,
            cuda_version=None,
            hip_version=None,
            reason="PyTorch is not installed.",
        )

    # ---------------------------------------------------------
    # PyTorch version information
    # ---------------------------------------------------------

    torch_version = getattr(torch, "__version__", None)

    torch_version_info = getattr(torch, "version", None)

    cuda_version = getattr(
        torch_version_info,
        "cuda",
        None,
    )

    hip_version = getattr(
        torch_version_info,
        "hip",
        None,
    )

    # ---------------------------------------------------------
    # GPU availability
    # ---------------------------------------------------------

    try:
        gpu_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return _base_result(
            backend="CPU",
            device="cpu",
            available=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            hip_version=hip_version,
            reason=f"GPU availability check failed: {exc}",
        )

    if not gpu_available:
        return _base_result(
            backend="CPU",
            device="cpu",
            available=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            hip_version=hip_version,
            reason="No usable GPU was detected.",
        )

    # ---------------------------------------------------------
    # GPU count
    # ---------------------------------------------------------

    try:
        gpu_count = int(torch.cuda.device_count())
    except Exception:
        gpu_count = 0

    if gpu_count <= 0:
        return _base_result(
            backend="CPU",
            device="cpu",
            available=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            hip_version=hip_version,
            reason="PyTorch reports GPU availability but no GPU devices.",
        )

    # ---------------------------------------------------------
    # Primary GPU
    # ---------------------------------------------------------

    device_index = 0
    device = f"cuda:{device_index}"

    try:
        properties = torch.cuda.get_device_properties(device_index)

        gpu_name = getattr(
            properties,
            "name",
            "Unknown GPU",
        )

        total_memory = getattr(
            properties,
            "total_memory",
            0,
        )

        vram_gb = (
            float(total_memory) / (1024 ** 3)
            if total_memory
            else None
        )

        # -----------------------------------------------------
        # NVIDIA compute capability
        # -----------------------------------------------------

        compute_capability = None

        major = getattr(
            properties,
            "major",
            None,
        )

        minor = getattr(
            properties,
            "minor",
            None,
        )

        if major is not None and minor is not None:
            compute_capability = f"{major}.{minor}"

        # -----------------------------------------------------
        # AMD architecture
        #
        # ROCm PyTorch commonly exposes:
        #
        #     properties.gcnArchName
        #
        # e.g.
        #
        #     gfx1201
        # -----------------------------------------------------

        architecture = getattr(
            properties,
            "gcnArchName",
            None,
        )

        # -----------------------------------------------------
        # Determine backend
        #
        # ROCm PyTorch exposes HIP through torch.version.hip.
        # Therefore HIP takes priority over CUDA detection.
        # -----------------------------------------------------

        if hip_version:
            return {
                "backend": "ROCm",
                "device": device,
                "available": True,
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "vram_gb": vram_gb,
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "compute_capability": None,
                "architecture": architecture,
                "reason": None,
            }

        # -----------------------------------------------------
        # NVIDIA CUDA
        # -----------------------------------------------------

        return {
            "backend": "CUDA",
            "device": device,
            "available": True,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "vram_gb": vram_gb,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "hip_version": None,
            "compute_capability": compute_capability,
            "architecture": None,
            "reason": None,
        }

    except Exception as exc:
        return _base_result(
            backend="CPU",
            device="cpu",
            available=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            hip_version=hip_version,
            reason=f"GPU detection failed: {exc}",
        )


def get_backend_name() -> str:
    """Return the detected backend name."""
    return detect_backend()["backend"]


def get_device() -> str:
    """Return the device string used by Ultralytics.

    Examples:

        CUDA  -> "cuda:0"
        ROCm  -> "cuda:0"
        CPU   -> "cpu"
    """
    return detect_backend()["device"]


def gpu_available() -> bool:
    """Return True when a usable GPU backend is available."""
    return bool(detect_backend()["available"])


if __name__ == "__main__":
    import pprint

    pprint.pprint(
        detect_backend(),
        sort_dicts=False,
    )