import streamlit as st
import json
import os

# Function to save settings to a JSON file
def save_settings(settings):
    with open("saved_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

def load_settings():
    if os.path.exists("saved_settings.json"):
        with open("saved_settings.json", "r") as f:
            return json.load(f)
    return {"root_location": "", "ultralytics_models_location": "", "preferred_backend": "Auto"}

def show_settings_tab():
    st.title("Settings")
    settings = load_settings()
    
    root_location = st.text_input("Default Root Location", value=settings.get("root_location", ""))
    ultralytics_models_location = st.text_input("Ultralytics Models Location", value=settings.get("ultralytics_models_location", ""))
    
    preferred_backend = st.selectbox(
        "Preferred Inference Backend",
        options=["Auto", "CUDA", "ROCm", "CPU"],
        index=["Auto", "CUDA", "ROCm", "CPU"].index(settings.get("preferred_backend", "Auto"))
    )
    
    if st.button("Save"):
        settings["root_location"] = root_location
        settings["ultralytics_models_location"] = ultralytics_models_location
        settings["preferred_backend"] = preferred_backend
        save_settings(settings)
        st.success("Settings saved successfully!")