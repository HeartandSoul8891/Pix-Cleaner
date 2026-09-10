"""
Pix-Cleaner - Ultralytics inference backend.

This module contains the actual YOLO inference logic used by Pix-Cleaner.

Hardware/backend detection is handled separately by:

    scripts.detect_backend

PyTorch CUDA and ROCm both expose their GPU through torch.cuda, so the
device passed to Ultralytics will normally be:

    NVIDIA CUDA -> "cuda:0"
    AMD ROCm    -> "cuda:0"
    CPU          -> "cpu"

This keeps the YOLO code hardware-independent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from scripts.detect_backend import detect_backend


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ----------------------------------------------------------------------
# Project / path helpers
# ----------------------------------------------------------------------

def _project_root() -> Path:
    """Return the Pix-Cleaner project root directory."""
    return Path(__file__).resolve().parent.parent


def init_root(root_location: str | Path) -> Path:
    """
    Resolve and create the configured root directory.

    Args:
        root_location: Root directory configured by the user.

    Returns:
        Resolved Path object.
    """
    root = Path(root_location).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_folders_in_root(root_location: str | Path) -> list[str]:
    """Return subdirectories available beneath the configured root."""
    root = init_root(root_location)

    return sorted(
        folder.name
        for folder in root.iterdir()
        if folder.is_dir()
    )

def list_models_in_location(
    models_location: str | Path,
) -> list[str]:
    """
    Return supported model files recursively from the configured
    Ultralytics model directory.

    Models may be organised in subdirectories, for example:

        ultralytics/
        ├── bbox/
        │   ├── model_a.pt
        │   └── model_b.pt
        └── sgm/
            ├── model_c.pt
            └── model_d.pt

    Returned paths are relative to models_location:

        bbox/model_a.pt
        bbox/model_b.pt
        sgm/model_c.pt
        sgm/model_d.pt

    Supported formats:

        .pt
        .onnx
        .engine
        .torchscript
        .xml
    """

    location = Path(models_location).expanduser()

    if not location.exists() or not location.is_dir():
        return []

    supported_extensions = {
        ".pt",
        ".onnx",
        ".engine",
        ".torchscript",
        ".xml",
    }

    models: list[str] = []

    for file in location.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix.lower() not in supported_extensions:
            continue

        # Return a path relative to the configured model directory.
        relative_path = file.relative_to(location)

        models.append(
            relative_path.as_posix()
        )

    return sorted(models)

# ----------------------------------------------------------------------
# Filter helpers
# ----------------------------------------------------------------------

def get_filter_dir(root_location: str | Path) -> Path:
    """Return the filter directory used by Pix-Cleaner."""
    return init_root(root_location) / "filters"


def list_filters(root_location: str | Path) -> list[str]:
    """Return available filter files."""
    filter_dir = get_filter_dir(root_location)

    if not filter_dir.exists():
        return []

    return sorted(
        file.name
        for file in filter_dir.iterdir()
        if file.is_file()
    )


def load_filter_classes(
    root_location: str | Path,
    filter_name: str,
) -> list[str]:
    """
    Load class names from a filter file.

    Empty lines and comment lines beginning with '#' are ignored.
    """
    filter_path = get_filter_dir(root_location) / filter_name

    if not filter_path.exists():
        return []

    classes: list[str] = []

    with filter_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            classes.append(line)

    return classes


def _supported_images(
    directory: str | Path,
) -> list[Path]:
    """Return supported images from a directory."""
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        return []

    return sorted(
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )


# ----------------------------------------------------------------------
# Model / class helpers
# ----------------------------------------------------------------------

def _normalise_model_names(
    names: Any,
) -> dict[int, str]:
    """Normalize Ultralytics model class names into {id: name}."""
    if names is None:
        return {}

    if isinstance(names, dict):
        return {
            int(key): str(value)
            for key, value in names.items()
        }

    if isinstance(names, (list, tuple)):
        return {
            index: str(value)
            for index, value in enumerate(names)
        }

    return {}


def get_model_names(
    model: YOLO,
) -> dict[int, str]:
    """Return class IDs and names from a loaded YOLO model."""
    names = getattr(model, "names", None)
    return _normalise_model_names(names)


def parse_target_classes(
    target_classes: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """
    Normalize target class input.

    Accepts:

        "person"
        "person, dog, cat"
        ["person", "dog"]

    Returns:
        Cleaned list of class names.
    """
    if target_classes is None:
        return []

    if isinstance(target_classes, str):
        values = target_classes.split(",")

    else:
        values = list(target_classes)

    result: list[str] = []

    for value in values:
        value = str(value).strip()

        if value and value not in result:
            result.append(value)

    return result


def model_metadata(
    model_path: str | Path,
) -> dict[str, Any]:
    """
    Load a YOLO model and return useful metadata.

    The model is loaded only for inspection; no inference is performed.
    """
    model = _load_model(model_path)

    names = get_model_names(model)

    return {
        "model": str(model_path),
        "classes": names,
        "class_count": len(names),
        "backend": detect_backend(),
    }


def get_class_ids(
    model: YOLO,
    target_classes: list[str],
) -> list[int]:
    """
    Convert class names into model class IDs.

    Raises:
        ValueError if one or more requested classes do not exist.
    """
    names = get_model_names(model)

    if not target_classes:
        return []

    name_to_id = {
        name.lower(): class_id
        for class_id, name in names.items()
    }

    class_ids: list[int] = []
    missing: list[str] = []

    for class_name in target_classes:
        class_id = name_to_id.get(class_name.lower())

        if class_id is None:
            missing.append(class_name)
        else:
            class_ids.append(class_id)

    if missing:
        available = ", ".join(names.values())

        raise ValueError(
            "Unknown target class(es): "
            f"{', '.join(missing)}. "
            f"Available classes: {available}"
        )

    return class_ids


def _validate_filter_match(
    model: YOLO,
    target_classes: list[str],
) -> list[int]:
    """Validate target classes and return their numeric IDs."""
    return get_class_ids(
        model=model,
        target_classes=target_classes,
    )


# ----------------------------------------------------------------------
# Backend / inference configuration
# ----------------------------------------------------------------------

def _get_device() -> str:
    """
    Return the device selected by Pix-Cleaner's backend detector.

    Examples:

        NVIDIA CUDA -> "cuda:0"
        AMD ROCm    -> "cuda:0"
        CPU          -> "cpu"
    """
    backend = detect_backend()
    return backend["device"]


def _prediction_kwargs(
    conf_threshold: float,
    imgsz: int,
    class_ids: list[int] | None = None,
    inference_mode: str = "auto",
) -> dict[str, Any]:
    """
    Build common Ultralytics prediction arguments.

    Device selection is deliberately handled here so every inference
    path uses the same backend configuration.
    """
    kwargs: dict[str, Any] = {
        "conf": float(conf_threshold),
        "imgsz": int(imgsz),
        "verbose": False,
        "device": _get_device(),
    }

    if class_ids:
        kwargs["classes"] = class_ids

    mode = str(inference_mode).strip().lower()

    if mode == "nms":
        kwargs["end2end"] = False

    elif mode == "nms_free":
        kwargs["nms"] = False

    elif mode == "auto":
        pass

    else:
        raise ValueError(
            "Unknown inference mode: "
            f"{inference_mode!r}. "
            "Expected 'auto', 'nms', or 'nms_free'."
        )

    return kwargs


def _load_model(
    model_path: str | Path,
) -> YOLO:
    """Load an Ultralytics YOLO model."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    return YOLO(str(model_path))


def predict_image(
    model: YOLO,
    image_path: str | Path,
    conf_threshold: float,
    imgsz: int,
    class_ids: list[int] | None = None,
    inference_mode: str = "auto",
):
    """
    Run YOLO inference on one image.

    The selected CUDA/ROCm/CPU device is passed explicitly to
    Ultralytics.
    """
    kwargs = _prediction_kwargs(
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    return model.predict(
        source=str(image_path),
        **kwargs,
    )


# ----------------------------------------------------------------------
# Detection extraction
# ----------------------------------------------------------------------

def _extract_detections(
    result: Any,
) -> list[dict[str, Any]]:
    """
    Extract detections from an Ultralytics result object.

    Returns a simple list containing:

        class_id
        class_name
        confidence
        xyxy
    """
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return detections

    names = _normalise_model_names(
        getattr(result, "names", None)
    )

    try:
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        coordinates = boxes.xyxy.cpu().numpy()

    except Exception:
        return detections

    for class_id, confidence, xyxy in zip(
        class_ids,
        confidences,
        coordinates,
    ):
        class_id = int(class_id)

        detections.append(
            {
                "class_id": class_id,
                "class_name": names.get(
                    class_id,
                    str(class_id),
                ),
                "confidence": float(confidence),
                "xyxy": [
                    float(value)
                    for value in xyxy
                ],
            }
        )

    return detections


# ----------------------------------------------------------------------
# Preview
# ----------------------------------------------------------------------

def preview_dataset(
    input_dir: str | Path,
    target_classes: str | list[str] | tuple[str, ...] | None,
    model_path: str | Path,
    conf_threshold: float,
    imgsz: int,
    sample_size: int,
    batch_size: int = 8,
    inference_mode: str = "auto",
):
    """
    Preview a sample of images using batched inference.

    Returns a list containing the original image path, annotated RGB preview,
    detections, and the backend information used for inference.
    """
    input_dir = Path(input_dir)

    images = _supported_images(input_dir)

    if not images:
        return []

    images = images[:max(0, int(sample_size))]
    batch_size = max(1, int(batch_size))

    model = _load_model(model_path)

    target_classes = parse_target_classes(
        target_classes
    )

    class_ids = _validate_filter_match(
        model=model,
        target_classes=target_classes,
    )

    backend = detect_backend()

    kwargs = _prediction_kwargs(
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    preview_results: list[dict[str, Any]] = []

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        sources = [str(image_path) for image_path in batch_images]

        results = model.predict(
            source=sources,
            batch=len(sources),
            **kwargs,
        )

        for image_path, result in zip(batch_images, results):
            detections = _extract_detections(result)

            annotated_image = result.plot(
                conf=True,
                labels=True,
                boxes=True,
            )

            # Ultralytics/OpenCV returns BGR; Streamlit expects RGB.
            annotated_image = annotated_image[..., ::-1]

            is_discard = bool(detections)

            preview_results.append(
                {
                    "image": image_path,
                    "filename": image_path.name,
                    "result": result,
                    "annotated_image": annotated_image,
                    "detections": detections,
                    "is_discard": is_discard,
                    "kept": is_discard,
                    "backend": backend,
                }
            )

    return preview_results


# ----------------------------------------------------------------------
# Full dataset sorting
# ----------------------------------------------------------------------

def sort_dataset(
    input_dir: str | Path,
    keep_dir: str | Path,
    discard_dir: str | Path,
    target_classes: str | list[str] | tuple[str, ...] | None,
    model_path: str | Path,
    conf_threshold: float,
    imgsz: int,
    inference_mode: str = "auto",
    batch_size: int = 8,
):
    """
    Run YOLO inference over a complete dataset and move images into
    keep/discard directories.

    Images containing at least one target detection are moved to
    discard_dir.

    Images without a target detection are moved to keep_dir.

    Batched inference is retained from the original implementation.
    The selected CUDA/ROCm/CPU device is now passed explicitly.
    """
    input_dir = Path(input_dir)
    keep_dir = Path(keep_dir)
    discard_dir = Path(discard_dir)

    keep_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    discard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = _supported_images(input_dir)

    if not images:
        return 0, 0

    model = _load_model(model_path)

    target_classes = parse_target_classes(
        target_classes
    )

    class_ids = _validate_filter_match(
        model=model,
        target_classes=target_classes,
    )

    batch_size = max(
        1,
        int(batch_size),
    )

    kwargs = _prediction_kwargs(
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    kept = 0
    discarded = 0

    for start in range(
        0,
        len(images),
        batch_size,
    ):
        batch_images = images[
            start:start + batch_size
        ]

        sources = [
            str(image_path)
            for image_path in batch_images
        ]

        results = model.predict(
            source=sources,
            batch=batch_size,
            **kwargs,
        )

        for image_path, result in zip(
            batch_images,
            results,
        ):
            detections = _extract_detections(
                result
            )

            destination = (
                discard_dir
                if detections
                else keep_dir
            )

            shutil.move(
                str(image_path),
                str(destination / image_path.name),
            )

            if detections:
                discarded += 1
            else:
                kept += 1

    return kept, discarded


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def model_diagnostics(
    model_path: str | Path,
    image_path: str | Path,
    conf_threshold: float,
    imgsz: int,
    target_classes: str | list[str] | tuple[str, ...] | None,
    inference_mode: str = "auto",
) -> dict[str, Any]:
    """
    Run inference on one image and return diagnostic information.

    This is intended for the diagnostics section of the UI.
    """
    model = _load_model(model_path)

    target_classes = parse_target_classes(
        target_classes
    )

    class_ids = _validate_filter_match(
        model=model,
        target_classes=target_classes,
    )

    backend = detect_backend()

    results = predict_image(
        model=model,
        image_path=image_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    detections: list[dict[str, Any]] = []

    if results:
        detections = _extract_detections(
            results[0]
        )

    return {
        "model": str(model_path),
        "image": str(image_path),
        "backend": backend,
        "device": backend["device"],
        "detections": detections,
        "detection_count": len(detections),
        "target_classes": target_classes,
        "class_ids": class_ids,
        "confidence_threshold": float(conf_threshold),
        "image_size": int(imgsz),
        "inference_mode": inference_mode,
    }