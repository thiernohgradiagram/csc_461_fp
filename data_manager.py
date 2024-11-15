import os
import kagglehub
import shutil
import hashlib

class DataManager:

    def __init__(self):
        pass
        

    def get_directory_size(self, path):
        """
        Calculates the total size of files in a directory.
        
        Parameters:
            path (str): Path to the directory.
        
        Returns:
            int: Total size in bytes of all files in the directory.
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def get_file_checksum(self, file_path):
        """
        Calculates the SHA-256 checksum of a file.
        
        Parameters:
            file_path (str): Path to the file.
        
        Returns:
            str: SHA-256 checksum of the file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compare_directories(self, source_path, destination_path):
        """
        Compares files in the source and destination directories by checksum.
        
        Parameters:
            source_path (str): Path to the source directory.
            destination_path (str): Path to the destination directory.
        
        Returns:
            bool: True if all files match, False otherwise.
        """
        for dirpath, _, filenames in os.walk(source_path):
            for filename in filenames:
                source_file = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(source_file, source_path)
                dest_file = os.path.join(destination_path, relative_path)
                
                # Check if the corresponding file exists in destination
                if not os.path.exists(dest_file):
                    return False
                
                # Compare file checksums
                if self.get_file_checksum(source_file) != self.get_file_checksum(dest_file):
                    return False
                
        # if the size of the source and destination directories are different, then they are not the same and return False
        if self.get_directory_size(source_path) != self.get_directory_size(destination_path):
            print("Directory sizes do not match.")
            return False
        
        return True

    
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
            if self.compare_directories(path_to_copy_from, destination_path):
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



