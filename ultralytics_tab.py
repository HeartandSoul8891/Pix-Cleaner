import streamlit as st
from pathlib import Path

from settings_tab import load_settings
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


# ============================================================
# DEFAULTS
# ============================================================

CONF_THRESHOLD = 0.35

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


# ============================================================
# MAIN TAB
# ============================================================

def show_ultralytics_tab():

    st.title("Cleaner (Ultralytics)")

    st.write(
        "Clean and divide datasets into cleaned and dirty "
        "folders using YOLO models."
    )

    settings = load_settings()

    root_location = settings.get(
        "root_location",
        ""
    )

    ultralytics_models_location = settings.get(
        "ultralytics_models_location",
        ""
    )

    # --------------------------------------------------------
    # SETTINGS VALIDATION
    # --------------------------------------------------------

    if not root_location:
        st.warning(
            "Please set a default root location "
            "in the Settings tab."
        )
        return

    if not ultralytics_models_location:
        st.warning(
            "Please set a default Ultralytics models "
            "location in the Settings tab."
        )
        return

    # --------------------------------------------------------
    # ROOT
    # --------------------------------------------------------

    if st.button("Check Root Path"):
        resolved_path = init_root(root_location)

        st.info(
            f"Active Root Directory: `{resolved_path}`"
        )

    # --------------------------------------------------------
    # SELECT FOLDER
    # --------------------------------------------------------

    folder_options = list_folders_in_root(
        root_location
    )

    selected_folder = st.selectbox(
        "Select a folder to analyze",
        [""] + folder_options
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    model_options = list_models_in_location(
        ultralytics_models_location
    )

    selected_model = st.selectbox(
        "Select an Ultralytics model",
        [""] + model_options
    )

    # --------------------------------------------------------
    # TARGET CLASS / MODEL METADATA
    # --------------------------------------------------------

    st.subheader("Detection Target")

    target_col, meta_col = st.columns([3, 1])

    with target_col:
        target_classes = st.text_input(
            "Target class(es)",
            placeholder="e.g. dick  •  or 0  •  or person, car",
            help=(
                "Enter model class names or IDs. Separate multiple targets "
                "with commas. Leave empty to use any detection from the model."
            ),
            key="cleaner_target_classes",
        )

    with meta_col:
        st.write("")
        st.write("")
        read_metadata = st.button("📋 Read Model Metadata")

    if read_metadata:
        if not selected_model:
            st.error("Select a model first.")
        else:
            try:
                metadata = model_metadata(
                    Path(ultralytics_models_location) / selected_model
                )
                st.session_state["cleaner_model_metadata"] = metadata
            except Exception as e:
                st.error("Could not read model metadata.")
                st.exception(e)

    metadata = st.session_state.get("cleaner_model_metadata")
    current_model_path = (Path(ultralytics_models_location) / selected_model).resolve() if selected_model else None
    if metadata and current_model_path and metadata.get("model_path") == str(current_model_path):
        st.write("### Model Classes")
        names = metadata.get("model_names", {})
        st.caption(
            f"{len(names)} class(es) • task: {metadata.get('task') or 'unknown'}"
        )
        if names:
            st.dataframe(
                [{"ID": cid, "Class": name} for cid, name in names.items()],
                hide_index=True,
                width="stretch",
            )

    # ========================================================
    # GPU / BATCH
    # ========================================================

    st.subheader("GPU / Batch")

    try:
        backend_info = detect_backend()
        backend_name = str(
            backend_info.get("backend", backend_info.get("name", "Unknown"))
        )
        device = str(backend_info.get("device", "unknown"))
        gpu_name = str(
            backend_info.get(
                "gpu_name",
                backend_info.get("device_name", ""),
            )
            or ""
        )

        gpu_active = device.lower().startswith("cuda")

        gpu_col, batch_col = st.columns(2)

        with gpu_col:
            if gpu_active:
                st.success("🟢 GPU acceleration available")
            else:
                st.warning("🟡 CPU inference")

            st.caption(
                f"Backend: `{backend_name}` • Device: `{device}`"
            )

            if gpu_name:
                st.caption(f"GPU: `{gpu_name}`")

        with batch_col:
            batch_size = st.select_slider(
                "Batch Size",
                options=[1, 2, 4, 8, 16, 32],
                value=8,
                help=(
                    "Number of images sent to YOLO at once. Higher values "
                    "can increase GPU utilization but require more VRAM. "
                    "Start at 8 and reduce it if you run out of VRAM."
                ),
            )

    except Exception as e:
        batch_size = 8
        st.warning(f"Could not read GPU backend status: {e}")

    st.caption(
        f"Batching: `{batch_size}` image(s) per inference call. "
        "The full dataset run uses this value."
    )

    # ========================================================
    # INFERENCE SETTINGS
    # ========================================================

    st.subheader("Inference Settings")

    col1, col2 = st.columns(2)

    with col1:
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.00,
            max_value=1.00,
            value=0.10,
            step=0.01,
        )

    with col2:
        img_size = st.select_slider(
            "Inference Image Size (px)",
            options=[
                320,
                480,
                640,
                800,
                1024,
                1280,
                1536,
                1920,
            ],
            value=640,
        )

    # --------------------------------------------------------
    # YOLO26 INFERENCE MODE
    # --------------------------------------------------------

    inference_mode = st.selectbox(
        "Inference Mode",
        options=[
            "auto",
            "nms",
            "nms_free",
        ],
        format_func=lambda mode: {
            "auto": "Auto — Ultralytics default",
            "nms": (
                "Traditional NMS — YOLO26 one-to-many"
            ),
            "nms_free": (
                "NMS-free — YOLO26 one-to-one"
            ),
        }[mode],
        index=0,
        help=(
            "Auto uses the normal Ultralytics inference path. "
            "Traditional NMS forces end2end=False. "
            "NMS-free explicitly uses nms=False."
        ),
    )

    st.caption(
        f"Current inference: "
        f"{img_size}px • "
        f"confidence ≥ {conf_threshold:.2f} • "
        f"mode = {inference_mode}"
    )

    st.divider()

    # ========================================================
    # PREVIEW
    # ========================================================

    st.subheader("Benchmark & Calibration")

    col_prev1, col_prev2 = st.columns([1, 2])

    with col_prev1:
        sample_size = st.number_input(
            "Sample size",
            min_value=1,
            max_value=100,
            value=32, # max size of the sample size for previewing
            step=1,
        )

    with col_prev2:
        st.write("")
        st.write("")

        run_preview = st.button(
            "🔍 Preview Sample (No Files Moved)"
        )

    if run_preview:
        if (
            selected_folder
            and selected_model
        ):
            input_path = (
                Path(root_location)
                / selected_folder
            )

            full_model_path = (
                Path(ultralytics_models_location)
                / selected_model
            )

            with st.spinner(
                "Generating sample detections..."
            ):
                try:
                    preview_data = preview_dataset(
                        input_dir=input_path,
                        target_classes=target_classes,
                        model_path=full_model_path,
                        conf_threshold=conf_threshold,
                        imgsz=img_size,
                        sample_size=int(sample_size),
                        batch_size=int(batch_size),
                        inference_mode=inference_mode,
                    )

                except Exception as e:
                    st.error("Inference failed.")
                    st.exception(e)
                    preview_data = None

            if preview_data is None:
                pass

            elif not preview_data:
                st.warning(
                    "No supported images found "
                    "in the selected folder."
                )

            else:
                st.write("### Preview Report")

                grid_cols = st.columns(3)

                for idx, item in enumerate(
                    preview_data
                ):
                    with grid_cols[idx % 3]:
                        st.image(
                            item.get("annotated_image", item["image"]),
                            caption=item["filename"],
                            width="stretch",
                        )

                        if item["is_discard"]:
                            st.error(
                                "❌ DISCARD / DIRTY"
                            )

                            detection_text = ", ".join(
                                f"{d['class_name']} ({d['confidence']:.2f})"
                                for d in item["detections"]
                            )
                            st.caption(
                                f"Detected: {detection_text}"
                            )

                        else:
                            st.success(
                                "✅ KEEP / CLEAN"
                            )

                            st.caption(
                                "No target classes detected."
                            )

        else:
            st.error(
                "Please select a folder, model, "
                "and target class (if needed) before generating a preview."
            )

    # ========================================================
    # MODEL DIAGNOSTICS
    # ========================================================

    st.divider()

    st.subheader("🔬 Model Diagnostics")

    st.caption(
        "Use this when a model appears to be blind. "
        "It does not move any files."
    )

    diagnostic_enabled = st.checkbox(
        "Enable single-image diagnostic",
        value=False,
    )

    if diagnostic_enabled:

        diagnostic_image = st.text_input(
            "Image path for diagnostic",
            placeholder=r"C:\path\to\test_image.jpg",
        )

        diagnostic_conf = st.slider(
            "Diagnostic Confidence",
            min_value=0.00,
            max_value=1.00,
            value=0.01,
            step=0.01,
            key="diagnostic_conf",
        )

        diagnostic_size = st.select_slider(
            "Diagnostic Image Size",
            options=[
                320,
                480,
                640,
                800,
                1024,
                1280,
                1536,
                1920,
            ],
            value=640,
            key="diagnostic_size",
        )

        if st.button("🧪 Run Model Diagnostic"):

            if not selected_model:
                st.error("Select a model first.")

            elif not diagnostic_image:
                st.error("Enter an image path first.")

            elif not Path(diagnostic_image).is_file():
                st.error(
                    "The diagnostic image does not exist."
                )

            else:
                full_model_path = (
                    Path(ultralytics_models_location)
                    / selected_model
                )

                try:
                    with st.spinner(
                        "Running diagnostic..."
                    ):
                        diagnostic = model_diagnostics(
                            model_path=full_model_path,
                            image_path=diagnostic_image,
                            conf_threshold=diagnostic_conf,
                            imgsz=diagnostic_size,
                            target_classes=target_classes,
                            inference_mode=inference_mode,
                        )

                    st.write("### Model Classes")

                    model_names = diagnostic[
                        "model_names"
                    ]

                    if model_names:
                        for class_id, name in model_names.items():
                            st.write(
                                f"**{class_id}** → `{name}`"
                            )
                    else:
                        st.warning(
                            "The model did not report "
                            "any class names."
                        )

                    st.write("### Target Mapping")

                    st.write(
                        "Target classes:",
                        diagnostic["target_classes"],
                    )

                    st.write(
                        "Resolved class IDs:",
                        diagnostic["class_ids"],
                    )

                    if diagnostic["missing_classes"]:
                        st.warning(
                            "These target classes were not found "
                            "in the model: "
                            + ", ".join(
                                diagnostic["missing_classes"]
                            )
                        )

                    st.write("### Inference")

                    st.write(
                        f"Resolution: `{diagnostic['imgsz']}px`"
                    )

                    st.write(
                        f"Confidence: "
                        f"`{diagnostic['conf_threshold']}`"
                    )

                    st.write(
                        f"Mode: "
                        f"`{diagnostic['inference_mode']}`"
                    )

                    st.write("### Detections")

                    detection_count = diagnostic[
                        "detection_count"
                    ]

                    if detection_count == 0:
                        st.error("❌ ZERO DETECTIONS")

                        st.info(
                            "Try confidence 0.01, 640px, "
                            "then compare Auto vs Traditional NMS."
                        )

                    else:
                        st.success(
                            f"✅ {detection_count} detection(s)"
                        )

                        for detection in diagnostic[
                            "detections"
                        ]:
                            st.write(
                                f"**{detection['class_name']}** "
                                f"(class {detection['class_id']}) "
                                f"— "
                                f"{detection['confidence']:.4f} "
                                f"— "
                                f"`{detection['xyxy']}`"
                            )

                except Exception as e:
                    st.error("Diagnostic failed.")
                    st.exception(e)

    # ========================================================
    # FULL DATASET
    # ========================================================

    st.divider()

    st.subheader("Run Full Dataset Batch")

    st.warning(
        "This will physically move images into "
        "'cleaned' or 'dirty'. Use Preview first."
    )

    if st.button("Analyze & Sort Entire Folder"):

        if (
            selected_folder
            and selected_model
        ):
            input_path = (
                Path(root_location)
                / selected_folder
            )

            keep_path = (
                input_path
                / "non-detection"
            )

            discard_path = (
                input_path
                / "detected"
            )

            full_model_path = (
                Path(ultralytics_models_location)
                / selected_model
            )

            with st.spinner(
                "Processing full dataset..."
            ):
                try:
                    kept, discarded = sort_dataset(
                        input_dir=input_path,
                        keep_dir=keep_path,
                        discard_dir=discard_path,
                        target_classes=target_classes,
                        model_path=full_model_path,
                        conf_threshold=conf_threshold,
                        imgsz=img_size,
                        batch_size=int(batch_size),
                        inference_mode=inference_mode,
                    )

                    st.success(
                        "Sorting complete! "
                        f"Moved {kept} images to 'non-detection' "
                        f"and {discarded} images to 'detected'."
                    )

                except Exception as e:
                    st.error("Dataset processing failed.")
                    st.exception(e)

        else:
            st.error(
                "Please select a folder, model, "
                "and target class (if needed) before proceeding."
            )