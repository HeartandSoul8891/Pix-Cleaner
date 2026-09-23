import streamlit as st
from pathlib import Path

from tab.settings_tab import load_settings
from scripts.detect_backend import detect_backend
from scripts.ultralytics_script import (
    init_root,
    list_folders_in_root,
    list_models_in_location,
    model_metadata,
    sort_dataset,
    preview_dataset,
    model_diagnostics,
)

CONF_THRESHOLD = 0.10
IMAGE_SIZES = [320, 480, 640, 800, 1024, 1280, 1536, 1920]


def _backend_section(key_prefix: str):
    info = detect_backend()
    device = str(info.get("device", "cpu"))
    active_gpu = device.lower().startswith("cuda")
    left, right = st.columns(2)
    with left:
        st.subheader("GPU / Backend")
        if active_gpu:
            st.success("🟢 GPU acceleration available")
        else:
            st.warning("🟡 CPU inference")
        st.caption(f"Backend: `{info.get('backend', 'Unknown')}` • Device: `{device}`")
        if info.get("gpu_name"):
            st.caption(f"GPU: `{info['gpu_name']}`")
    with right:
        st.subheader("Batch")
        default_batch = 8 if active_gpu else 2
        batch = st.select_slider(
            "Images per inference call",
            options=[1, 2, 4, 8, 16, 32],
            value=default_batch,
            key=f"{key_prefix}_batch",
            help="Increase this to improve GPU utilisation; reduce it if VRAM is exhausted.",
        )
    return int(batch)

def _metadata_panel(model_path: Path, key_prefix: str):
    # Give the button a unique key so it doesn't collide with our data storage
    if st.button("📋 Read Model Metadata", key=f"{key_prefix}_metadata_btn"):
        try:
            # Store the dictionary under a separate session state key
            st.session_state[f"{key_prefix}_metadata_dict"] = model_metadata(model_path)
        except Exception as exc:
            st.error("Could not read model metadata.")
            st.exception(exc)

    # Retrieve from the dedicated metadata storage key
    metadata = st.session_state.get(f"{key_prefix}_metadata_dict")
    if not isinstance(metadata, dict) or metadata.get("model_path") != str(model_path.resolve()):
        return

    names = metadata.get("model_names", {})
    st.write("### Model Metadata")
    a, b, c = st.columns(3)
    a.metric("Task", metadata.get("model_type", metadata.get("task", "unknown")))
    b.metric("Classes", len(names))
    c.metric("Backend", metadata.get("backend", {}).get("backend", "unknown"))
    if names:
        st.dataframe(
            [{"ID": cid, "Class": name} for cid, name in names.items()],
            hide_index=True,
            width="stretch",
        )

def _model_selector(settings, key_prefix: str):
    location_key = "bbox_models_location" if key_prefix == "bbox" else "segm_models_location"
    location = settings.get(location_key, "")
    models = list_models_in_location(location) if location else []
    selected = st.selectbox(
        "Select model",
        [""] + models,
        key=f"{key_prefix}_model",
        help=f"Models are scanned recursively from the configured {key_prefix} model folder.",
    )
    return Path(location) / selected if location and selected else None


def _common_controls(key_prefix: str, batch: int):
    left, right = st.columns(2)
    with left:
        conf = st.slider("Confidence Threshold", 0.00, 1.00, CONF_THRESHOLD, 0.01, key=f"{key_prefix}_conf")
    with right:
        imgsz = st.select_slider("Inference Image Size", IMAGE_SIZES, 640, key=f"{key_prefix}_imgsz")
    mode = st.selectbox(
        "Inference Mode",
        ["auto", "nms", "nms_free"],
        format_func=lambda m: {"auto": "Auto", "nms": "Traditional NMS", "nms_free": "NMS-free"}[m],
        key=f"{key_prefix}_mode",
    )
    target = st.text_input(
        "Target class(es)",
        placeholder="person, face, open_eye",
        key=f"{key_prefix}_targets",
        help="Comma-separated class names or numeric class IDs. Leave empty to treat any model result as a match.",
    )
    return float(conf), int(imgsz), mode, target, batch


def _preview_and_run(settings, key_prefix, task, selected_model, selected_folder, conf, imgsz, mode, target, batch):
    root = settings.get("root_location", "")
    if not root or not selected_folder or selected_model is None:
        st.info("Select a dataset folder and a model first.")
        return
    input_path = Path(root) / selected_folder
    st.subheader("Preview & Calibration")
    sample = st.number_input("Sample size", 1, 100, 32, 1, key=f"{key_prefix}_sample")
    if st.button("🔍 Preview Sample (No Files Moved)", key=f"{key_prefix}_preview"):
        try:
            with st.spinner("Running inference..."):
                data = preview_dataset(input_path, target, selected_model, conf, imgsz, sample, batch, mode, task)
            if not data:
                st.warning("No supported images were found in the selected folder.")
            else:
                cols = st.columns(3)
                for i, item in enumerate(data):
                    with cols[i % 3]:
                        st.image(item["annotated_image"], caption=item["filename"], width="stretch")
                        if item["is_discard"]:
                            st.error("MATCH / DETECTED")
                        else:
                            st.success("NO MATCH")
                        if item["detections"]:
                            st.caption(", ".join(f"{d['class_name']} ({d['confidence']:.2f})" for d in item["detections"]))
        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)

    st.divider()
    st.subheader("Full Dataset")
    st.warning("This moves images out of the selected input folder. Use Preview first.")
    detected_name = st.text_input("Detected folder name", value=f"{Path(selected_model).stem}_detected", key=f"{key_prefix}_detected_name")
    clean_name = st.text_input("No-detection folder name", value=f"{Path(selected_model).stem}_no-detection", key=f"{key_prefix}_clean_name")
    if st.button("▶ Run Full Dataset", type="primary", key=f"{key_prefix}_run"):
        try:
            detected_dir = input_path / detected_name
            clean_dir = input_path / clean_name
            with st.spinner("Sorting dataset..."):
                kept, discarded = sort_dataset(input_path, clean_dir, detected_dir, target, selected_model, conf, imgsz, mode, batch, task)
            st.success(f"Finished. No-detection: {kept} • Detected: {discarded}")
        except Exception as exc:
            st.error("Dataset sorting failed.")
            st.exception(exc)


def _diagnostics(selected_model, key_prefix, task, target, conf, imgsz, mode):
    st.divider()
    st.subheader("🔬 Single-image Diagnostic")
    image = st.text_input("Image path", key=f"{key_prefix}_diagnostic_image")
    diagnostic_conf = st.slider("Diagnostic confidence", 0.00, 1.00, 0.01, 0.01, key=f"{key_prefix}_diagnostic_conf")
    if st.button("🧪 Run Diagnostic", key=f"{key_prefix}_diagnostic"):
        if selected_model is None or not image or not Path(image).is_file():
            st.error("Select a model and enter an existing image path.")
            return
        try:
            result = model_diagnostics(selected_model, image, diagnostic_conf, imgsz, target, mode, task)
            if result.get("annotated_image") is not None:
                st.image(result["annotated_image"], width="stretch")
            st.write("### Results")
            st.json({k: v for k, v in result.items() if k not in {"annotated_image"}})
        except Exception as exc:
            st.error("Diagnostic failed.")
            st.exception(exc)


def show_ultralytics_tab():
    st.title("Cleaner — Bounding Box")
    st.caption("Detection models using bounding boxes. CUDA, ROCm and CPU use the same inference layer.")
    settings = load_settings()
    if not settings.get("root_location") or not settings.get("bbox_models_location"):
        st.warning("Configure the dataset root and BBox model folder in Settings first.")
        return
    folders = list_folders_in_root(settings["root_location"])
    selected_folder = st.selectbox("Dataset folder", [""] + folders, key="bbox_folder")
    selected_model = _model_selector(settings, "bbox")
    if selected_model:
        _metadata_panel(selected_model, "bbox")
    batch = _backend_section("bbox")
    conf, imgsz, mode, target, batch = _common_controls("bbox", batch)
    _preview_and_run(settings, "bbox", "detect", selected_model, selected_folder, conf, imgsz, mode, target, batch)
    _diagnostics(selected_model, "bbox", "detect", target, conf, imgsz, mode)
