from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from scripts.detect_backend import detect_backend

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript", ".xml"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def init_root(root_location: str | Path) -> Path:
    root = Path(root_location).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_folders_in_root(root_location: str | Path) -> list[str]:
    root = init_root(root_location)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_models_in_location(models_location: str | Path) -> list[str]:
    location = Path(models_location).expanduser()
    if not location.is_dir():
        return []
    return sorted(
        p.relative_to(location).as_posix()
        for p in location.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS
    )


def _supported_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _normalise_model_names(names: Any) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def get_model_names(model: YOLO) -> dict[int, str]:
    return _normalise_model_names(getattr(model, "names", None))


def parse_target_classes(target_classes: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if target_classes is None:
        return []
    values = target_classes.split(",") if isinstance(target_classes, str) else list(target_classes)
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return result


def _load_model(model_path: str | Path) -> YOLO:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return YOLO(str(path))


def _model_task(model: YOLO) -> str:
    task = getattr(model, "task", None)
    if task:
        return str(task)
    overrides = getattr(model, "overrides", None) or {}
    return str(overrides.get("task") or "unknown")


def model_metadata(model_path: str | Path) -> dict[str, Any]:
    """Load a model once and return the keys expected by the UI.

    The previous implementation returned ``classes`` while the UI looked for
    ``model_names`` and ``model_path``. This mismatch was the reason the
    metadata display silently stopped matching the loaded model.
    """
    path = Path(model_path).expanduser().resolve()
    model = _load_model(path)
    names = get_model_names(model)
    task = _model_task(model)
    backend = detect_backend()

    return {
        "model_path": str(path),
        "model": str(path),
        "model_names": names,
        "classes": names,
        "class_count": len(names),
        "task": task,
        "backend": backend,
        "model_type": "Segmentation" if task == "segment" else "Detection" if task == "detect" else task,
    }


def get_class_ids(model: YOLO, target_classes: list[str]) -> list[int]:
    names = get_model_names(model)
    if not target_classes:
        return []
    name_to_id = {name.lower(): class_id for class_id, name in names.items()}
    ids: list[int] = []
    missing: list[str] = []
    for value in target_classes:
        if value.isdigit() and int(value) in names:
            ids.append(int(value))
            continue
        class_id = name_to_id.get(value.lower())
        if class_id is None:
            missing.append(value)
        else:
            ids.append(class_id)
    if missing:
        raise ValueError(
            f"Unknown target class(es): {', '.join(missing)}. "
            f"Available classes: {', '.join(names.values())}"
        )
    return ids


def _prediction_kwargs(conf_threshold: float, imgsz: int, class_ids: list[int] | None = None, inference_mode: str = "auto") -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "conf": float(conf_threshold),
        "imgsz": int(imgsz),
        "verbose": False,
        "device": detect_backend()["device"],
    }
    if class_ids:
        kwargs["classes"] = class_ids
    mode = str(inference_mode).strip().lower()
    if mode == "nms":
        kwargs["end2end"] = False
    elif mode == "nms_free":
        kwargs["nms"] = False
    elif mode != "auto":
        raise ValueError("inference_mode must be 'auto', 'nms', or 'nms_free'")
    return kwargs


def _extract_detections(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    names = _normalise_model_names(getattr(result, "names", None))
    try:
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        coordinates = boxes.xyxy.cpu().numpy()
    except Exception:
        return []
    return [
        {
            "class_id": int(class_id),
            "class_name": names.get(int(class_id), str(int(class_id))),
            "confidence": float(confidence),
            "xyxy": [float(v) for v in xyxy],
        }
        for class_id, confidence, xyxy in zip(class_ids, confidences, coordinates)
    ]


def _extract_segments(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None:
        return []
    names = _normalise_model_names(getattr(result, "names", None))
    try:
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        coordinates = boxes.xyxy.cpu().numpy()
        polygons = getattr(masks, "xy", None) or []
    except Exception:
        return []

    segments: list[dict[str, Any]] = []
    for index, (class_id, confidence, xyxy) in enumerate(zip(class_ids, confidences, coordinates)):
        polygon = polygons[index] if index < len(polygons) else None
        segments.append(
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(int(class_id))),
                "confidence": float(confidence),
                "xyxy": [float(v) for v in xyxy],
                "polygon": np.asarray(polygon).tolist() if polygon is not None else [],
            }
        )
    return segments


def _extract_task_items(result: Any, task: str) -> list[dict[str, Any]]:
    return _extract_segments(result) if task == "segment" else _extract_detections(result)


def _annotated_image(result: Any) -> np.ndarray:
    image = result.plot(conf=True, labels=True, boxes=True, masks=True)
    return image[..., ::-1]


def preview_dataset(input_dir, target_classes, model_path, conf_threshold, imgsz, sample_size, batch_size=8, inference_mode="auto", task="detect"):
    images = _supported_images(input_dir)[:max(0, int(sample_size))]
    if not images:
        return []
    batch_size = max(1, int(batch_size))
    model = _load_model(model_path)
    actual_task = _model_task(model)
    if task == "segment" and actual_task != "segment":
        raise ValueError(f"Selected model is not a segmentation model (task={actual_task!r}).")
    if task == "detect" and actual_task not in {"detect", "unknown"}:
        raise ValueError(f"Selected model is not a bounding-box detection model (task={actual_task!r}).")
    target_classes = parse_target_classes(target_classes)
    class_ids = get_class_ids(model, target_classes)
    kwargs = _prediction_kwargs(conf_threshold, imgsz, class_ids, inference_mode)
    backend = detect_backend()
    output: list[dict[str, Any]] = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        results = model.predict(source=[str(p) for p in batch], batch=len(batch), **kwargs)
        for image_path, result in zip(batch, results):
            items = _extract_task_items(result, task)
            output.append({
                "image": image_path,
                "filename": image_path.name,
                "result": result,
                "annotated_image": _annotated_image(result),
                "detections": items,
                "is_discard": bool(items),
                "kept": not bool(items),
                "backend": backend,
                "task": task,
            })
    return output


def sort_dataset(input_dir, keep_dir, discard_dir, target_classes, model_path, conf_threshold, imgsz, inference_mode="auto", batch_size=8, task="detect"):
    input_dir = Path(input_dir)
    keep_dir = Path(keep_dir)
    discard_dir = Path(discard_dir)
    keep_dir.mkdir(parents=True, exist_ok=True)
    discard_dir.mkdir(parents=True, exist_ok=True)
    images = _supported_images(input_dir)
    if not images:
        return 0, 0
    model = _load_model(model_path)
    actual_task = _model_task(model)
    if task == "segment" and actual_task != "segment":
        raise ValueError(f"Selected model is not a segmentation model (task={actual_task!r}).")
    if task == "detect" and actual_task not in {"detect", "unknown"}:
        raise ValueError(f"Selected model is not a bounding-box detection model (task={actual_task!r}).")
    class_ids = get_class_ids(model, parse_target_classes(target_classes))
    kwargs = _prediction_kwargs(conf_threshold, imgsz, class_ids, inference_mode)
    batch_size = max(1, int(batch_size))
    kept = discarded = 0
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        results = model.predict(source=[str(p) for p in batch], batch=len(batch), **kwargs)
        for image_path, result in zip(batch, results):
            detected = bool(_extract_task_items(result, task))
            destination = discard_dir if detected else keep_dir
            shutil.move(str(image_path), str(destination / image_path.name))
            if detected:
                discarded += 1
            else:
                kept += 1
    return kept, discarded


def model_diagnostics(model_path, image_path, conf_threshold, imgsz, target_classes, inference_mode="auto", task="detect"):
    model = _load_model(model_path)
    names = get_model_names(model)
    class_ids = get_class_ids(model, parse_target_classes(target_classes))
    results = model.predict(source=str(image_path), **_prediction_kwargs(conf_threshold, imgsz, class_ids, inference_mode))
    result = results[0] if results else None
    items = _extract_task_items(result, task) if result is not None else []
    return {
        "model": str(model_path),
        "model_path": str(Path(model_path).resolve()),
        "model_names": names,
        "task": _model_task(model),
        "image": str(image_path),
        "backend": detect_backend(),
        "device": detect_backend()["device"],
        "detections": items,
        "detection_count": len(items),
        "target_classes": parse_target_classes(target_classes),
        "class_ids": class_ids,
        "missing_classes": [],
        "confidence_threshold": float(conf_threshold),
        "conf_threshold": float(conf_threshold),
        "image_size": int(imgsz),
        "imgsz": int(imgsz),
        "inference_mode": inference_mode,
        "annotated_image": _annotated_image(result) if result is not None else None,
    }


def run_detector_on_folder(input_dir, output_detected_dir, output_clean_dir, model_path, target_classes, conf_threshold, imgsz, batch_size=8, inference_mode="auto", task="detect"):
    """Pipeline-friendly wrapper. Does not require the input folder to be the project root."""
    return sort_dataset(
        input_dir=input_dir,
        keep_dir=output_clean_dir,
        discard_dir=output_detected_dir,
        target_classes=target_classes,
        model_path=model_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        inference_mode=inference_mode,
        batch_size=batch_size,
        task=task,
    )
