import os
import numpy as np
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
        # Dictionary to hold extracted features and labels
        data = {
            "features": [],
            "label": []
        }

        genres = os.listdir(dataset_path)  # Assuming each genre is a folder

        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith(".wav"):  # Process only .au files
                        file_path = os.path.join(genre_path, file)

                        try:
                            y, sr = self.load_audio(file_path)
                            features = self.extract_features_from_file(y, sr)
                            data["features"].append(features)
                            data["label"].append(genre)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")

        # return the extracted features and labels
        return data

