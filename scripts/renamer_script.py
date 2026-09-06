import os

def list_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def rename_files(folder_path, new_name, use_folder_name=False):
    files = list_files_in_folder(folder_path)
    num_files = len(files)
    num_digits = len(str(num_files))
    
    renamed_files = []
    
    for i, file in enumerate(files):
        if use_folder_name:
            folder_name = os.path.basename(folder_path)
            new_filename = f"{folder_name}_{str(i+1).zfill(num_digits)}{os.path.splitext(file)[1]}"
        else:
            new_filename = f"{new_name}_{str(i+1).zfill(num_digits)}{os.path.splitext(file)[1]}"
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_filename))
        renamed_files.append(new_filename)
    
    return renamed_files