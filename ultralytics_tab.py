import streamlit as st
from pathlib import Path
from settings_tab import load_settings
from scripts.ultralytics_script import (
    init_root,
    list_folders_in_root,
    list_models_in_location,
    list_filters,
    get_filter_dir,
    sort_dataset,
    preview_dataset
)

CONF_THRESHOLD = 0.35
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def show_ultralytics_tab():
    st.title("Cleaner (Ultralytics)")
    st.write("Clean and divide datasets into cleaned and dirty folders using YOLO models.")
    
    settings = load_settings()
    root_location = settings.get("root_location", "")
    ultralytics_models_location = settings.get("ultralytics_models_location", "")
    
    if not root_location:
        st.warning("Please set a default root location in the Settings tab.")
        return
    if not ultralytics_models_location:
        st.warning("Please set a default Ultralytics models location in the Settings tab.")
        return
    
    if st.button("Check Root Path"):
        resolved_path = init_root(root_location)
        st.info(f"Active Root Directory: `{resolved_path}`")
    
    folder_options = list_folders_in_root(root_location)
    selected_folder = st.selectbox("Select a folder to analyze", [""] + folder_options)
    
    model_options = list_models_in_location(ultralytics_models_location)
    selected_model = st.selectbox("Select an Ultralytics model", [""] + model_options)
    
    filter_options = list_filters()
    selected_filter = st.selectbox("Select a filter", [""] + filter_options)
    
    if selected_filter:
        st.caption(f"Filter Path: `{get_filter_dir() / selected_filter}`")
    
    conf_threshold = st.slider("Confidence Threshold", 0.00, 1.00, 0.10, 0.01)
    img_size = st.select_slider("Inference Image Size (px)", options=[640, 800, 1024, 1280], value=1024)
    
    st.divider()

    # --- PREVIEW SECTION ---
    st.subheader("Benchmark & Calibration")
    col_prev1, col_prev2 = st.columns([1, 2])
    with col_prev1:
        sample_size = st.number_input("Sample size", min_value=1, max_value=30, value=6)
    with col_prev2:
        st.write("")
        st.write("")
        run_preview = st.button("🔍 Preview Sample (No Files Moved)")

    if run_preview:
        if selected_folder and selected_model and selected_filter:
            input_path = Path(root_location) / selected_folder
            full_model_path = Path(ultralytics_models_location) / selected_model
            filter_file_path = get_filter_dir() / selected_filter

            with st.spinner("Generating sample detections..."):
                preview_data = preview_dataset(
                    input_dir=input_path,
                    filter_file_path=filter_file_path,
                    model_path=full_model_path,
                    conf_threshold=conf_threshold,  # Pass conf_threshold as an argument
                    sample_size=sample_size
                )

            if not preview_data:
                st.warning("No supported images found in the selected folder.")
            else:
                st.write("### Preview Report")
                # Render in a 3-column grid
                grid_cols = st.columns(3)
                for idx, item in enumerate(preview_data):
                    with grid_cols[idx % 3]:
                        st.image(item["image"], caption=item["filename"], use_container_width=True)
                        if item["is_discard"]:
                            st.error("❌ DISCARD / DIRTY")
                            st.caption(f"Detected: {', '.join(item['detections'])}")
                        else:
                            st.success("✅ KEEP / CLEAN")
                            st.caption("No target classes detected.")
        else:
            st.error("Please select a folder, model, and filter before generating a preview.")

    st.divider()

    # --- EXECUTION SECTION ---
    st.subheader("Run Full Dataset Batch")
    if st.button("Analyze & Sort Entire Folder"):
        if selected_folder and selected_model and selected_filter:
            input_path = Path(root_location) / selected_folder
            keep_path = input_path / "cleaned"
            discard_path = input_path / "dirty"
            full_model_path = Path(ultralytics_models_location) / selected_model
            filter_file_path = get_filter_dir() / selected_filter

            with st.spinner("Processing full dataset..."):
                kept, discarded = sort_dataset(
                    input_dir=input_path,
                    keep_dir=keep_path,
                    discard_dir=discard_path,
                    filter_file_path=filter_file_path,
                    model_path=full_model_path,
                    conf_threshold=conf_threshold
                )
            st.success(f"Sorting complete! Moved {kept} images to 'cleaned' and {discarded} images to 'dirty'.")
        else:
            st.error("Please select a folder, model, and filter before proceeding.")