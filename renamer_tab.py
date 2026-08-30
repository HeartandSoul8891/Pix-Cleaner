import streamlit as st
import os as os
from settings_tab import load_settings
from scripts.renamer_script import list_files_in_folder, rename_files

# Function to display the renamer tab
def show_renamer_tab():
    st.title("Renamer")
    st.write("This tab allows you to rename files and folders.")
    
    # Load settings
    settings = load_settings()
    root_location = settings.get("root_location", "")
    
    if not root_location:
        st.warning("Please set a default root location in the Settings tab.")
        return
    
    # Dropdown to select a folder
    folder_options = [f for f in os.listdir(root_location) if os.path.isdir(os.path.join(root_location, f))]
    selected_folder = st.selectbox("Select a folder", [""] + folder_options)
    
    if selected_folder:
        folder_path = os.path.join(root_location, selected_folder)
        
        # Button to read files
        if st.button("Read"):
            files = list_files_in_folder(folder_path)
            st.text_area("Preview", "\n".join(files), height=150)
        
        # Field to define a new name for the files
        new_name = st.text_input("New Name for Files")
        
        # Checkbox for folder name + number function
        use_folder_name_and_number = st.checkbox("Use folder name + number")
        
        # Button to change file names
        if st.button("Change File Names"):
            if use_folder_name_and_number:
                renamed_files = rename_files(folder_path, use_folder_name_and_number)
                st.success("Files renamed successfully!")
                st.text_area("Renamed Preview", "\n".join(renamed_files), height=150)
            elif new_name:
                renamed_files = rename_files(folder_path, new_name, use_folder_name_and_number)
                st.success("Files renamed successfully!")
                st.text_area("Renamed Preview", "\n".join(renamed_files), height=150)
            else:
                st.error("Please enter a new name for the files or use the folder name + number option.")
    else:
        st.warning("Please select a folder.")