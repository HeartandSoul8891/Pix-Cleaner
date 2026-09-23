import json
from pathlib import Path

import streamlit as st

SETTINGS_FILE = Path(__file__).resolve().parent / "saved_settings.json"

DEFAULT_SETTINGS = {
    "root_location": "",
    "bbox_models_location": "",
    "segm_models_location": "",
    # Backward compatibility with the previous setting name.
    "ultralytics_models_location": "",
    "preferred_backend": "Auto",
}


def save_settings(settings: dict) -> None:
    data = dict(DEFAULT_SETTINGS)
    data.update(settings)
    SETTINGS_FILE.write_text(
        json.dumps(data, indent=4),
        encoding="utf-8",
    )


def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)

    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)

    settings = dict(DEFAULT_SETTINGS)
    settings.update(data)

    # Existing installations used one shared Ultralytics directory. Keep it
    # working by using it as the default for both task-specific locations.
    legacy = settings.get("ultralytics_models_location", "")
    if not settings.get("bbox_models_location"):
        settings["bbox_models_location"] = legacy
    if not settings.get("segm_models_location"):
        settings["segm_models_location"] = legacy

    return settings


def _path_status(label: str, value: str) -> None:
    if not value:
        st.caption(f"{label}: not configured")
        return
    path = Path(value).expanduser()
    if path.is_dir():
        st.caption(f"{label}: ✅ `{path}`")
    else:
        st.caption(f"{label}: ⚠️ folder does not currently exist")


def show_settings_tab() -> None:
    st.title("Settings")
    st.caption("Configure the dataset root and separate Ultralytics model libraries.")

    settings = load_settings()

    root_location = st.text_input(
        "Default Root Location",
        value=settings.get("root_location", ""),
        help="The root containing the folders that Pix-Cleaner can process.",
    )

    st.subheader("Ultralytics model folders")

    bbox_models_location = st.text_input(
        "Bounding-box Models Location",
        value=settings.get("bbox_models_location", ""),
        help="Folder containing detection/bounding-box models. Subfolders are scanned recursively.",
    )
    _path_status("BBox models", bbox_models_location)

    segm_models_location = st.text_input(
        "Segmentation Models Location",
        value=settings.get("segm_models_location", ""),
        help="Folder containing segmentation models. Subfolders are scanned recursively.",
    )
    _path_status("Segmentation models", segm_models_location)

    st.subheader("Inference backend")
    backends = ["Auto", "CUDA", "ROCm", "CPU"]
    current_backend = settings.get("preferred_backend", "Auto")
    backend_index = backends.index(current_backend) if current_backend in backends else 0
    preferred_backend = st.selectbox(
        "Preferred Inference Backend",
        options=backends,
        index=backend_index,
        help="Auto uses the existing backend detector. The preference is stored for the inference layer.",
    )

    if st.button("💾 Save Settings", type="primary"):
        settings.update(
            {
                "root_location": root_location.strip(),
                "bbox_models_location": bbox_models_location.strip(),
                "segm_models_location": segm_models_location.strip(),
                # Keep the old key populated for older code/plugins.
                "ultralytics_models_location": bbox_models_location.strip(),
                "preferred_backend": preferred_backend,
            }
        )
        save_settings(settings)
        st.success("Settings saved.")
        st.rerun()
