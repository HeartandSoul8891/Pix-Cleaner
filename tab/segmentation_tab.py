import streamlit as st
from pathlib import Path

from tab.settings_tab import load_settings
from tab.bbox_tab import (
    _backend_section,
    _common_controls,
    _metadata_panel,
    _preview_and_run,
    _diagnostics,
)
from scripts.ultralytics_script import list_folders_in_root, list_models_in_location


def show_segmentation_tab():
    st.title("Cleaner — Segmentation")
    st.caption("Segmentation models with mask previews, class metadata, diagnostics and dataset sorting.")
    settings = load_settings()
    if not settings.get("root_location") or not settings.get("segm_models_location"):
        st.warning("Configure the dataset root and Segmentation model folder in Settings first.")
        return

    folders = list_folders_in_root(settings["root_location"])
    selected_folder = st.selectbox("Dataset folder", [""] + folders, key="segm_folder")
    models = list_models_in_location(settings["segm_models_location"])
    selected = st.selectbox("Select segmentation model", [""] + models, key="segm_model")
    selected_model = Path(settings["segm_models_location"]) / selected if selected else None

    if selected_model:
        _metadata_panel(selected_model, "segm")
    batch = _backend_section("segm")
    conf, imgsz, mode, target, batch = _common_controls("segm", batch)
    _preview_and_run(settings, "segm", "segment", selected_model, selected_folder, conf, imgsz, mode, target, batch)
    _diagnostics(selected_model, "segm", "segment", target, conf, imgsz, mode)
