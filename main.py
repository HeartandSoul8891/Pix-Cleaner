import streamlit as st
from settings_tab import show_settings_tab
from renamer_tab import show_renamer_tab
from ultralytics_tab import show_ultralytics_tab

# Main function to run the Streamlit app
def main():
    st.title("File Management App")
    
    tab = st.tabs(["Cleaner (Ultralytics)", "Renamer", "Settings"])
    
    with tab[0]:
        show_ultralytics_tab()
    with tab[1]:
        show_renamer_tab()
    with tab[2]:
        show_settings_tab()

if __name__ == "__main__":
    main()

    