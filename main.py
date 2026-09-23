import streamlit as st

from tab.settings_tab import show_settings_tab
from tab.renamer_tab import show_renamer_tab
from tab.bbox_tab import show_ultralytics_tab
from tab.segmentation_tab import show_segmentation_tab
from tab.pipeline_tab import show_pipeline_tab


def main():
    tabs = st.tabs([
        "🔍 BBox Cleaner",
        "🎭 Segmentation Cleaner",
        "🔗 Multi-Detector Pipeline",
        "🔠 Renamer",
        "💡 Settings",
    ])
    with tabs[0]:
        show_ultralytics_tab()
    with tabs[1]:
        show_segmentation_tab()
    with tabs[2]:
        show_pipeline_tab()
    with tabs[3]:
        show_renamer_tab()
    with tabs[4]:
        show_settings_tab()


if __name__ == "__main__":
    main()
