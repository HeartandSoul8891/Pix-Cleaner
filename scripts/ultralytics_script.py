from pathlib import Path
from ultralytics import YOLO

CONF_THRESHOLD = 0.35
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def load_filter_classes(filter_file_path):
    """Load filter classes from a file."""
    filter_classes = []
    filter_path = Path(filter_file_path)
    if filter_path.is_file():
        with open(filter_path, 'r') as file:
            for line in file:
                if line.strip():
                    filter_classes.append(line.strip())
    return filter_classes

def init_root(root_location):
    """Initialize the root directory and create necessary subdirectories."""
    root_path = Path(root_location)
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path

def list_folders_in_root(root_location):
    """List all folders in the root directory."""
    root_path = Path(root_location)
    if not root_path.exists():
        return []
    return [f.name for f in root_path.iterdir() if f.is_dir()]

def list_models_in_location(models_location):
    """List all YOLO models in the specified location."""
    models_path = Path(models_location)
    if not models_path.exists():
        return []
    return [f.name for f in models_path.iterdir() if f.is_file()]

def get_filter_dir():
    """Get the directory containing filter files."""
    return Path("filters")

def list_filters():
    """List all filter files."""
    filter_dir = get_filter_dir()
    if not filter_dir.exists():
        return []
    return [f.name for f in filter_dir.iterdir() if f.is_file()]

def get_class_ids(model, filter_classes):
    """Map string filter names to model class IDs."""
    if not filter_classes or not hasattr(model, "names"):
        return None
    return [cid for cid, name in model.names.items() if name in filter_classes]

def sort_dataset(input_dir, keep_dir, discard_dir, filter_file_path, model_path, conf_threshold=CONF_THRESHOLD, imgsz=1024):
    """Sorts the dataset into 'cleaned' and 'dirty' folders based on inference results."""
    input_dir = Path(input_dir)
    keep_dir = Path(keep_dir)
    discard_dir = Path(discard_dir)
    filter_classes = load_filter_classes(filter_file_path)

    model = YOLO(str(model_path))
    class_ids = get_class_ids(model, filter_classes)

    kept = 0
    discarded = 0

    for path in input_dir.glob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            results = model.predict(
                source=str(path), 
                conf=conf_threshold, 
                imgsz=imgsz,
                classes=class_ids,
                verbose=False
            )[0]
            
            has_detections = len(results.boxes) > 0
            
            if has_detections:
                (discard_dir / path.name).parent.mkdir(parents=True, exist_ok=True)
                path.rename(discard_dir / path.name)
                discarded += 1
            else:
                (keep_dir / path.name).parent.mkdir(parents=True, exist_ok=True)
                path.rename(keep_dir / path.name)
                kept += 1

    return kept, discarded

def preview_dataset(input_dir, filter_file_path, model_path, conf_threshold=CONF_THRESHOLD, sample_size=6, imgsz=1024):
    """Runs non-destructive inference on a small sample of images for calibration."""
    input_dir = Path(input_dir)
    filter_classes = load_filter_classes(filter_file_path)

    model = YOLO(str(model_path))
    class_ids = get_class_ids(model, filter_classes)

    image_paths = [p for p in input_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS][:sample_size]

    if not image_paths:
        return []

    preview_results = []

    for path in image_paths:
        results = model.predict(
            source=str(path), 
            conf=conf_threshold, 
            imgsz=imgsz,
            classes=class_ids,
            verbose=False
        )[0]
        
        annotated_bgr = results.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        
        has_detections = len(results.boxes) > 0
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)
            detections.append(f"{cls_name} ({conf:.2f})")
        
        preview_results.append({
            "filename": path.name,
            "image": annotated_rgb,
            "is_discard": has_detections,
            "detections": detections
        })

    return preview_results