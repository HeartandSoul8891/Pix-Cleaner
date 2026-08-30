import streamlit as st
from settings_tab import show_settings_tab
from renamer_tab import show_renamer_tab
from ultralytics_tab import show_ultralytics_tab

# Function to display the folder explorer tab
def folder_explorer_tab():
    st.title("Folder Explorer")
    st.write("This tab allows you to explore folders and files.")
    # Add your folder exploration logic here

# Main function to run the Streamlit app
def main():
    st.title("File Management App")
    
    tab = st.tabs(["Folder Explorer", "Cleaner (Ultralytics)", "Renamer", "Settings"])
    
    with tab[0]:
        folder_explorer_tab()
    with tab[1]:
        show_ultralytics_tab()
    with tab[2]:
        show_renamer_tab()
    with tab[3]:
        show_settings_tab()

if __name__ == "__main__":
    main()