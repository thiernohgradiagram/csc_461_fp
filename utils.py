

import hashlib
import os
import matplotlib.pyplot as plt
import numpy as np


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


def detect_outliers_iqr_and_plot(data, summary_stats, label_column):
    """
    Detects outliers for each feature using the IQR method and generates a boxplot.

    Parameters:
        data (pd.DataFrame): Dataset containing features for outlier detection.
        summary_stats (pd.DataFrame): Summary statistics including '25%' and '75%' for each feature.
        label_column (str): The name of the label column to exclude from analysis.
    """
    # Exclude the label column from analysis
    feature_columns = [col for col in data.columns if col != label_column]
    
    for feature in feature_columns:
        # Extract the feature data and summary statistics
        feature_data = data[feature]
        Q1 = summary_stats.loc['25%', feature]  # 25th percentile (Q1)
        Q3 = summary_stats.loc['75%', feature]  # 75th percentile (Q3)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outliers
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers_iqr = feature_data[(feature_data < lower_bound_iqr) | (feature_data > upper_bound_iqr)]

        # Plot the boxplot and highlight outliers, only if there are any outliers
        if len(outliers_iqr) > 0:
            print(f"Outliers detected for feature '{feature}':")
            print(outliers_iqr)
            
            # Plot the boxplot and highlight outliers
            plt.figure(figsize=(6, 4))
            plt.boxplot(feature_data, vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            plt.scatter(outliers_iqr, [1] * len(outliers_iqr), color='red', label='Outliers (IQR)')
            plt.title(f"Boxplot for {feature} with Outliers Highlighted")
            plt.xlabel("Values")
            plt.legend()
            plt.tight_layout()
            plt.show()

def handle_outliers(data, summary_stats, strategy='remove', label_column=None):
    """
    Handles outliers in the dataset using the specified strategy.
    
    Parameters:
        data (pd.DataFrame): Dataset containing features.
        summary_stats (pd.DataFrame): Summary statistics including '25%' and '75%' for each feature.
        strategy (str): Strategy to handle outliers ('remove', 'cap', 'transform', 'impute').
        label_column (str): Name of the label column to exclude.
        
    Returns:
        pd.DataFrame: Dataset after handling outliers.
    """
    # Exclude label column from processing
    feature_columns = [col for col in data.columns if col != label_column]
    data_processed = data.copy()

    for feature in feature_columns:
        Q1 = summary_stats.loc['25%', feature]
        Q3 = summary_stats.loc['75%', feature]
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if strategy == 'remove':
            data_processed = data_processed[
                (data_processed[feature] >= lower_bound) & (data_processed[feature] <= upper_bound)
            ]
        elif strategy == 'cap':
            data_processed[feature] = np.clip(data_processed[feature], lower_bound, upper_bound)
        elif strategy == 'transform':
            data_processed[feature] = np.log1p(data_processed[feature])
        elif strategy == 'impute':
            data_processed.loc[data_processed[feature] < lower_bound, feature] = data[feature].median()
            data_processed.loc[data_processed[feature] > upper_bound, feature] = data[feature].median()
    
    return data_processed

        
        
