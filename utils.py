

import hashlib
import os


def get_directory_size(path):
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

def get_file_checksum(file_path):
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

def compare_directories(source_path, destination_path):
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
            if get_file_checksum(source_file) != get_file_checksum(dest_file):
                return False
            
    # if the size of the source and destination directories are different, then they are not the same and return False
    if get_directory_size(source_path) != get_directory_size(destination_path):
        print("Directory sizes do not match.")
        return False
    
    return True