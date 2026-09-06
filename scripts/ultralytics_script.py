"""
Ultralytics inference / dataset-cleaning backend.

This module is designed to match ultralytics_tab.py.

Key behaviours:
- Supports single-class YOLO models even when the filter name differs
  from the model's internal class name (e.g. filter "dick", model "item").
- Supports multi-class filters by case-insensitive class-name matching.
- Supports YOLO26 inference modes:
    auto      -> normal Ultralytics prediction
    nms       -> end2end=False
    nms_free  -> nms=False
- Never passes an ambiguous empty classes=[] to Ultralytics.
- Provides preview, full sorting and single-image diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import shutil

import numpy as np
from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# PATH / FILE HELPERS
# ============================================================

def _project_root() -> Path:
    """Return the Pix-Cleaner project root."""
    return Path(__file__).resolve().parent.parent


def init_root(root_location: str | Path) -> Path:
    """
    Resolve and create the configured root directory.

    The directory is created when it does not yet exist.
    """
    root = Path(root_location).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_folders_in_root(root_location: str | Path) -> list[str]:
    """Return direct child directories of the configured root."""
    root = Path(root_location).expanduser()

    if not root.exists() or not root.is_dir():
        return []

    folders = [
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and p.name.lower() not in {"cleaned", "dirty"}
    ]

    return sorted(folders, key=str.lower)


def list_models_in_location(models_location: str | Path) -> list[str]:
    """
    Return supported model files in the configured model directory.

    Recursive search is used so model files can also live in subfolders.
    """
    root = Path(models_location).expanduser()

    if not root.exists() or not root.is_dir():
        return []

    extensions = {".pt", ".onnx", ".engine", ".torchscript", ".xml"}

    models = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    return sorted(models, key=str.lower)


def get_filter_dir() -> Path:
    """
    Return the filter directory.

    Preferred location:
        <Pix-Cleaner project>/filters

    The directory is created automatically.
    """
    filter_dir = _project_root() / "filters"
    filter_dir.mkdir(parents=True, exist_ok=True)
    return filter_dir


def list_filters() -> list[str]:
    """Return .txt filter files from the filter directory."""
    filter_dir = get_filter_dir()

    return sorted(
        [
            p.name
            for p in filter_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        ],
        key=str.lower,
    )


def load_filter_classes(filter_file_path: str | Path) -> list[str]:
    """
    Read filter classes from a text file.

    Empty lines and lines beginning with # are ignored.
    Duplicate entries are removed while preserving order.
    """
    path = Path(filter_file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Filter file not found: {path}")

    classes: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            value = raw_line.strip()

            if not value or value.startswith("#"):
                continue

            key = value.lower()

            if key not in seen:
                seen.add(key)
                classes.append(value)

    return classes


def _supported_images(input_dir: str | Path) -> list[Path]:
    """Return supported image files directly inside input_dir."""
    directory = Path(input_dir)

    if not directory.exists() or not directory.is_dir():
        return []

    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


# ============================================================
# MODEL / CLASS HELPERS
# ============================================================

def _normalise_model_names(model: Any) -> dict[int, str]:
    """Return model.names as a predictable {int: str} dictionary."""
    names = getattr(model, "names", {}) or {}

    if isinstance(names, list):
        return {
            int(index): str(name)
            for index, name in enumerate(names)
        }

    return {
        int(class_id): str(name)
        for class_id, name in names.items()
    }


def get_model_names(model: Any) -> dict[int, str]:
    """Public helper for retrieving normalized model class names."""
    return _normalise_model_names(model)



def parse_target_classes(target_classes):
    """Parse class names or numeric IDs entered by the user."""
    if target_classes is None:
        return []
    if isinstance(target_classes, str):
        raw = target_classes.replace(";", ",").replace("\n", ",").split(",")
    else:
        raw = []
        for value in target_classes:
            raw.extend(str(value).replace(";", ",").split(","))
    result, seen = [], set()
    for value in raw:
        value = str(value).strip()
        if value and value.lower() not in seen:
            result.append(value)
            seen.add(value.lower())
    return result


def model_metadata(model_path):
    """Load a model and return class/task metadata without inference."""
    model = _load_model(model_path)
    names = _normalise_model_names(model)
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_type": type(model).__name__,
        "task": getattr(model, "task", None),
        "class_count": len(names),
        "model_names": names,
    }

def get_class_ids(model, target_classes):
    """Resolve user-entered class names or IDs to model class IDs."""
    targets = parse_target_classes(target_classes)
    if not targets:
        return None
    names = _normalise_model_names(model)
    if not names:
        raise ValueError("The selected model does not contain any class names.")
    if len(names) == 1:
        return [next(iter(names.keys()))]
    lookup = {x.lower() for x in targets}
    return [int(cid) for cid, name in names.items()
            if str(cid).lower() in lookup or name.strip().lower() in lookup]


def _validate_filter_match(model, target_classes):
    """Validate and resolve user-entered target classes."""
    names = _normalise_model_names(model)
    targets = parse_target_classes(target_classes)
    if not names:
        raise ValueError("The selected model does not contain any class names.")
    if not targets:
        return {"matched": True, "class_ids": None, "model_names": names,
                "missing_classes": [], "message": "All model classes are enabled."}
    if len(names) == 1:
        cid = next(iter(names.keys()))
        return {"matched": True, "class_ids": [int(cid)], "model_names": names,
                "missing_classes": [], "message": f"Single-class model: {cid} = {names[cid]}."}
    lookup = {x.lower() for x in targets}
    matched, matched_keys = [], set()
    for cid, name in names.items():
        if str(cid).lower() in lookup or name.strip().lower() in lookup:
            matched.append(int(cid)); matched_keys.update({str(cid).lower(), name.strip().lower()})
    missing = [x for x in targets if x.lower() not in matched_keys]
    if not matched:
        raise ValueError("None of the entered target classes match the selected model.\n\n"
                         f"Targets: {targets}\nModel classes: {names}")
    return {"matched": True, "class_ids": matched, "model_names": names,
            "missing_classes": missing, "message": f"Matched target classes: {matched}"}


# ============================================================
# INFERENCE
# ============================================================

def _prediction_kwargs(
    conf_threshold: float,
    imgsz: int,
    class_ids: list[int] | None,
    inference_mode: str,
) -> dict[str, Any]:
    """
    Build the prediction kwargs.

    classes is only included when there are actual class IDs.
    This deliberately avoids passing classes=[].
    """
    kwargs: dict[str, Any] = {
        "conf": float(conf_threshold),
        "imgsz": int(imgsz),
        "verbose": False,
    }

    if class_ids is not None:
        if not class_ids:
            raise ValueError(
                "No valid model class IDs were resolved from the filter."
            )
        kwargs["classes"] = class_ids

    mode = str(inference_mode).lower().strip()

    if mode == "auto":
        pass
    elif mode == "nms":
        # YOLO26 traditional one-to-many + NMS path.
        kwargs["end2end"] = False
    elif mode == "nms_free":
        # YOLO26 one-to-one NMS-free path.
        kwargs["nms"] = False
    else:
        raise ValueError(
            f"Unknown inference mode: {inference_mode!r}. "
            "Use auto, nms or nms_free."
        )

    return kwargs


def _load_model(model_path: str | Path) -> YOLO:
    """Load and validate a YOLO model."""
    path = Path(model_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    return YOLO(str(path))


def predict_image(
    model: YOLO,
    image_path: str | Path,
    conf_threshold: float = 0.10,
    imgsz: int = 640,
    class_ids: list[int] | None = None,
    inference_mode: str = "auto",
):
    """
    Run inference on one image and return the first Ultralytics result.
    """
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    kwargs = _prediction_kwargs(
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    results = model.predict(
        source=str(image_path),
        **kwargs,
    )

    if not results:
        raise RuntimeError(
            "Ultralytics returned no result for the image."
        )

    return results[0]


def _extract_detections(
    result: Any,
    model_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Convert an Ultralytics result into simple Python dictionaries."""
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return detections

    for box in boxes:
        cls_tensor = getattr(box, "cls", None)
        conf_tensor = getattr(box, "conf", None)
        xyxy_tensor = getattr(box, "xyxy", None)

        if cls_tensor is None or conf_tensor is None or xyxy_tensor is None:
            continue

        class_id = int(cls_tensor.item())
        confidence = float(conf_tensor.item())

        xyxy = [
            round(float(value), 2)
            for value in xyxy_tensor[0].tolist()
        ]

        detections.append(
            {
                "class_id": class_id,
                "class_name": model_names.get(
                    class_id,
                    str(class_id),
                ),
                "confidence": confidence,
                "xyxy": xyxy,
            }
        )

    return detections


# ============================================================
# PREVIEW
# ============================================================

def preview_dataset(
    input_dir: str | Path,
    target_classes: str | list[str] | None,
    model_path: str | Path,
    conf_threshold: float = 0.10,
    imgsz: int = 640,
    sample_size: int = 6,
    inference_mode: str = "auto",
) -> list[dict[str, Any]]:
    """
    Run inference on a small sample without moving any files.

    Returns dictionaries containing:
        filename
        image
        is_discard
        detections
    """
    input_dir = Path(input_dir)
    images = _supported_images(input_dir)

    if not images:
        return []

    sample_size = max(1, int(sample_size))
    images = images[:sample_size]

    target_classes = parse_target_classes(target_classes)
    model = _load_model(model_path)
    model_names = _normalise_model_names(model)

    validation = _validate_filter_match(
        model,
        target_classes,
    )

    class_ids = validation["class_ids"]

    preview_data: list[dict[str, Any]] = []

    for image_path in images:
        result = predict_image(
            model=model,
            image_path=image_path,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            class_ids=class_ids,
            inference_mode=inference_mode,
        )

        detections = _extract_detections(
            result,
            model_names,
        )

        # Plot only the detections returned by the configured inference.
        plotted = result.plot()

        # Ultralytics returns a numpy array for plot().
        # Convert BGR -> RGB for Streamlit.
        if isinstance(plotted, np.ndarray):
            if plotted.ndim == 3 and plotted.shape[2] == 3:
                plotted = plotted[:, :, ::-1]

        preview_data.append(
            {
                "filename": image_path.name,
                "image": plotted,
                "is_discard": len(detections) > 0,
                "detections": [
                    (
                        f"{d['class_name']} "
                        f"({d['confidence']:.3f})"
                    )
                    for d in detections
                ],
            }
        )

    return preview_data


# ============================================================
# FULL DATASET SORTING
# ============================================================

def _safe_move(
    source: Path,
    destination_dir: Path,
) -> None:
    """
    Move one file into destination_dir.

    If a file with the same name already exists, generate a unique name
    instead of overwriting an existing image.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / source.name

    if destination.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1

        while True:
            candidate = (
                destination_dir
                / f"{stem}_{counter}{suffix}"
            )

            if not candidate.exists():
                destination = candidate
                break

            counter += 1

    shutil.move(str(source), str(destination))


def sort_dataset(
    input_dir: str | Path,
    keep_dir: str | Path,
    discard_dir: str | Path,
    target_classes: str | list[str] | None,
    model_path: str | Path,
    conf_threshold: float = 0.10,
    imgsz: int = 640,
    inference_mode: str = "auto",
) -> tuple[int, int]:
    """
    Analyze every supported image in input_dir.

    Behaviour:
        detection found -> dirty/discard
        no detection     -> cleaned/keep

    Returns:
        (kept_count, discarded_count)
    """
    input_dir = Path(input_dir)
    keep_dir = Path(keep_dir)
    discard_dir = Path(discard_dir)
    images = _supported_images(input_dir)

    if not images:
        return 0, 0

    target_classes = parse_target_classes(target_classes)
    model = _load_model(model_path)
    model_names = _normalise_model_names(model)

    validation = _validate_filter_match(
        model,
        target_classes,
    )

    class_ids = validation["class_ids"]

    kept = 0
    discarded = 0

    for image_path in images:
        result = predict_image(
            model=model,
            image_path=image_path,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            class_ids=class_ids,
            inference_mode=inference_mode,
        )

        boxes = getattr(result, "boxes", None)
        has_detection = (
            boxes is not None
            and len(boxes) > 0
        )

        if has_detection:
            _safe_move(image_path, discard_dir)
            discarded += 1
        else:
            _safe_move(image_path, keep_dir)
            kept += 1

    return kept, discarded


# ============================================================
# DIAGNOSTICS
# ============================================================

def model_diagnostics(
    model_path: str | Path,
    image_path: str | Path,
    conf_threshold: float = 0.01,
    imgsz: int = 640,
    target_classes: str | list[str] | None = None,
    inference_mode: str = "auto",
) -> dict[str, Any]:
    """
    Run a detailed single-image diagnostic.

    This never moves files.
    """
    model = _load_model(model_path)
    model_names = _normalise_model_names(model)

    targets = parse_target_classes(target_classes)
    validation = _validate_filter_match(model, targets)
    class_ids = validation["class_ids"]
    missing_classes = validation.get("missing_classes", [])

    result = predict_image(
        model=model,
        image_path=image_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        class_ids=class_ids,
        inference_mode=inference_mode,
    )

    detections = _extract_detections(
        result,
        model_names,
    )

    return {
        "model_path": str(Path(model_path).resolve()),
        "image_path": str(Path(image_path).resolve()),
        "model_names": model_names,
        "target_classes": targets,
        "class_ids": class_ids,
        "missing_classes": missing_classes,
        "imgsz": int(imgsz),
        "conf_threshold": float(conf_threshold),
        "inference_mode": inference_mode,
        "detection_count": len(detections),
        "detections": detections,
    }
