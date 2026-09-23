"""Multi-detector sequential pipeline UI for Pix-Cleaner."""

from pathlib import Path

import streamlit as st

from tab.settings_tab import load_settings
from scripts.ultralytics_script import list_models_in_location, run_detector_on_folder
from scripts.detect_backend import detect_backend

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _model_library(settings, task):
    key = "bbox_models_location" if task == "detect" else "segm_models_location"
    return settings.get(key, "")


def _models(settings, task):
    location = _model_library(settings, task)
    return list_models_in_location(location) if location else []


def _step_defaults(index):
    return {
        "task": "detect",
        "model": "",
        "source": "Original dataset" if index == 0 else "Previous detected branch",
        "target": "",
        "conf": 0.10,
        "imgsz": 640,
        "batch": 8,
    }


def _ensure_steps():
    if "pix_pipeline_steps" not in st.session_state:
        st.session_state["pix_pipeline_steps"] = [_step_defaults(0)]
    return st.session_state["pix_pipeline_steps"]


def _branch_name(step_index, model_name, matched):
    stem = Path(model_name).stem if model_name else f"step{step_index + 1}"
    return f"{stem}_{'detected' if matched else 'no-detection'}"


def show_pipeline_tab():
    st.title("Multi-Detector Pipeline")
    st.caption("Run multiple BBox and/or Segmentation models as a branching sequence. Each step creates a detected and a no-detection folder that can feed later steps.")
    settings = load_settings()
    root = settings.get("root_location", "")
    if not root:
        st.warning("Configure the Default Root Location in Settings first.")
        return

    steps = _ensure_steps()
    folders = sorted(p.name for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []
    base = st.selectbox("Pipeline starting folder", [""] + folders, key="pipeline_base")

    st.subheader("Pipeline steps")
    st.info("Example: Face detector → choose detected branch → Eye detector → choose open-eye branch. Each step can use either a BBox or Segmentation model.")

    for i, step in enumerate(steps):
        with st.expander(f"Step {i + 1}: {Path(step['model']).stem if step['model'] else 'not configured'}", expanded=True):
            task = st.radio("Detector type", ["detect", "segment"], index=0 if step["task"] == "detect" else 1, horizontal=True, key=f"pipe_task_{i}")
            step["task"] = task
            models = _models(settings, task)
            current_model = step.get("model", "")
            if current_model not in models:
                current_model = ""
            model = st.selectbox("Model", [""] + models, index=(models.index(current_model) + 1 if current_model else 0), key=f"pipe_model_{i}")
            step["model"] = model

            available_sources = ["Original dataset"]
            if i > 0:
                for previous in range(i):
                    prev_model = steps[previous].get("model") or f"step{previous + 1}"
                    available_sources.extend([
                        _branch_name(previous, prev_model, True),
                        _branch_name(previous, prev_model, False),
                    ])
            current_source = step.get("source", available_sources[0])
            if current_source not in available_sources:
                current_source = available_sources[0]
            step["source"] = st.selectbox("Input branch", available_sources, index=available_sources.index(current_source), key=f"pipe_source_{i}")

            c1, c2, c3 = st.columns(3)
            with c1:
                step["target"] = st.text_input("Target classes", value=step.get("target", ""), key=f"pipe_target_{i}")
            with c2:
                step["conf"] = st.slider("Confidence", 0.00, 1.00, float(step.get("conf", 0.10)), 0.01, key=f"pipe_conf_{i}")
            with c3:
                step["batch"] = st.select_slider("Batch", [1, 2, 4, 8, 16, 32], value=int(step.get("batch", 8)), key=f"pipe_batch_{i}")
            step["imgsz"] = st.select_slider("Image Size", [320, 480, 640, 800, 1024, 1280, 1536, 1920], value=int(step.get("imgsz", 640)), key=f"pipe_imgsz_{i}")

    a, b = st.columns(2)
    with a:
        if st.button("➕ Add Detector Step"):
            steps.append(_step_defaults(len(steps)))
            st.rerun()
    with b:
        if len(steps) > 1 and st.button("➖ Remove Last Step"):
            steps.pop()
            st.rerun()

    if st.button("🧹 Reset Pipeline"):
        st.session_state["pix_pipeline_steps"] = [_step_defaults(0)]
        st.rerun()

    st.divider()
    st.subheader("Run Pipeline")
    if st.button("▶ Run Multi-Detector Pipeline", type="primary"):
        if not base:
            st.error("Select a starting folder.")
            return
        if not steps or any(not s.get("model") for s in steps):
            st.error("Every pipeline step needs a model.")
            return

        # Resolve each step's source dynamically from the branches produced by earlier steps.
        branches = {"Original dataset": Path(root) / base}
        summary = []
        try:
            for i, step in enumerate(steps):
                source_dir = branches.get(step["source"])
                if source_dir is None:
                    raise FileNotFoundError(f"Pipeline source branch does not exist: {step['source']}")
                if not source_dir.is_dir():
                    raise FileNotFoundError(f"Pipeline input folder does not exist: {source_dir}")

                library = Path(_model_library(settings, step["task"]))
                model_path = library / step["model"]
                detected_name = _branch_name(i, step["model"], True)
                clean_name = _branch_name(i, step["model"], False)
                detected_dir = Path(root) / detected_name
                clean_dir = Path(root) / clean_name

                with st.spinner(f"Step {i + 1}: {Path(step['model']).stem}..."):
                    kept, detected = run_detector_on_folder(
                        source_dir,
                        detected_dir,
                        clean_dir,
                        model_path,
                        step["target"],
                        step["conf"],
                        step["imgsz"],
                        step["batch"],
                        "auto",
                        step["task"],
                    )
                branches[detected_name] = detected_dir
                branches[clean_name] = clean_dir
                summary.append({"step": i + 1, "model": step["model"], "detected": detected, "no_detection": kept})

            st.success("Pipeline completed.")
            st.dataframe(summary, hide_index=True, width="stretch")
        except Exception as exc:
            st.error("Pipeline failed. Earlier steps may already have moved files.")
            st.exception(exc)
