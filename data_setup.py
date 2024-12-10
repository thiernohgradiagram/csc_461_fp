import os
import kagglehub
import shutil
from utils import compare_directories

class DataSetup:

    def __init__(self):
        pass
    
    def download_data(self, dataset_name):
        """
        Downloads the dataset, forcing a re-download if the initial attempt is empty.

        Returns:
            str: The path to the dataset files.
        """
        # Attempt initial download
        path = kagglehub.dataset_download(dataset_name, force_download=False)

        # Check if download is empty and force a re-download if necessary
        if os.path.isdir(path) and not os.listdir(path):
            print("Downloaded directory is empty. Forcing re-download...")
            path = kagglehub.dataset_download(dataset_name, force_download=True)
        return path

    
    def move_data(self, path_to_copy_from, destination_path):
        """
        Moves files from one directory to another. If data already exists
        in the destination but doesn't match the source by checksum, it will
        re-copy the files.
        
        Parameters:
            path_to_copy_from (str): Path to the source directory.
            destination_path (str): Path to the destination directory.
        """
        # Check if destination path exists and validate files
        if os.path.exists(destination_path):
            if  compare_directories(path_to_copy_from, destination_path):
                print("Data already exists in the repo and matches source. Skipping copy.")
                return
            else:
                print("Data exists but does not match. Replacing data.")
                shutil.rmtree(destination_path)  # Remove mismatched data

        # Create destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # Move all files from source to destination
        for item in os.listdir(path_to_copy_from):
            s = os.path.join(path_to_copy_from, item)
            d = os.path.join(destination_path, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.move(s, d)
        
        print("Data moved successfully :)")

