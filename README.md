A streamlit cleaning utility that uses supports Ultralytics models to do the heavy lifting

To install, you'll need at least python 3.12 and git

copy the repo locally
in the local copy:

activate a shell and run: python -m venv venv
activate the venv

pip install -r requirements.txt

afterwards you can always start the app by "start.bat" or activate the venv and run "streamlit run main.py"
the app is pretty straight forward:
Cleaner -> cleaning
Renamer -> rename the contens of your file
Settings -> set the location where you have stored your ultralytics models and your root folder of images you want to clean.

<img width="608" height="669" alt="Schermafbeelding 2026-09-06 182812" src="https://github.com/user-attachments/assets/0e9f4a61-c6ca-4742-a434-94875aa2b6f7" />

"Check Root Path" initilises the root folder of your images
then play around with the settings

<img width="599" height="501" alt="Schermafbeelding 2026-09-06 182840" src="https://github.com/user-attachments/assets/3243eee3-871d-4209-ad0f-2375b11baf1d" />
the "Sample size" supports up to 30 samples of your selected folder
Model Diagnostics -> check if your yolo model is doing anything
Run Full Dataset -> start cleaning based on your settings
