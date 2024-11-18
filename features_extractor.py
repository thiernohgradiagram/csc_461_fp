import os
import numpy as np
import pandas as pd
import librosa

class FeaturesExtractor:

    def __init__(self):
        pass

    def load_audio(self, file_path):
        """
        Load an audio file.

        Parameters:
        - file_path (str): Path to the audio file.

        Returns:
        - y (np.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.
        """
        return librosa.load(file_path, sr=None)

    def extract_features_from_file(self, y, sr):
        """
        Extract features from an audio signal.

        Parameters:
        - y (np.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.

        Returns:
        - features (np.ndarray): Extracted feature vector.
        """
        # Time-Domain Features
        rms = np.mean(librosa.feature.rms(y=y))  # Root Mean Square Energy
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # Zero-Crossing Rate
        
        # Frequency-Domain Features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Cepstral Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)  # Average MFCCs for all frames
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features into a single vector
        features = np.hstack([
            rms, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
            mfccs_mean, chroma_mean
        ])
        return features

    def extract_features_all_files(self, dataset_path):
        """
        Process the dataset to extract features and save them to a CSV file.
        """

        # create a dictionary for the genre labels
        genre_dict = {
            'blues': 0,
            'classical': 1,
            'country': 2,
            'disco': 3,
            'hiphop': 4,
            'jazz': 5,
            'metal': 6,
            'pop': 7,
            'reggae': 8,
            'rock': 9
        }

        # Dictionary to hold extracted features and labels
        data = []

        genres = os.listdir(dataset_path)  # Assuming each genre is a folder

        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith(".wav"):  # Process only .wav files
                        file_path = os.path.join(genre_path, file)

                        try:
                            y, sr = self.load_audio(file_path)
                            features = self.extract_features_from_file(y, sr)
                            genre_num = genre_dict[genre]
                            data.append(np.hstack([features, genre_num]))
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")

        # Assign meaningful feature names
        column_names = [
            "RMS", "ZCR", "Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff",
            "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5", "MFCC_6", "MFCC_7",
            "MFCC_8", "MFCC_9", "MFCC_10", "MFCC_11", "MFCC_12", "MFCC_13",
            "Chroma_1", "Chroma_2", "Chroma_3", "Chroma_4", "Chroma_5", "Chroma_6",
            "Chroma_7", "Chroma_8", "Chroma_9", "Chroma_10", "Chroma_11", "Chroma_12",
            "Genre"
        ]

        # Convert to a DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # Return the DataFrame
        return df
