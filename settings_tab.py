import streamlit as st
import json
import os

# Function to load settings from a JSON file
def load_settings():
    if os.path.exists("saved_settings.json"):
        with open("saved_settings.json", "r") as f:
            return json.load(f)
    return {"root_location": "", "ultralytics_models_location": ""}

# Function to save settings to a JSON file
def save_settings(settings):
    with open("saved_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

# Function to display the settings tab
def show_settings_tab():
    st.title("Settings")
    st.write("This tab allows you to configure settings.")
    
    # Load existing settings
    settings = load_settings()
    
    # Field for setting a default location for all files (root)
    root_location = st.text_input("Default Root Location", value=settings.get("root_location", ""))
    
    # Field for setting a default location for Ultralytics models
    ultralytics_models_location = st.text_input("Ultralytics Models Location", value=settings.get("ultralytics_models_location", ""))
    
    # Save button
    if st.button("Save"):
        settings["root_location"] = root_location
        settings["ultralytics_models_location"] = ultralytics_models_location
        save_settings(settings)
        st.success("Settings saved successfully!")