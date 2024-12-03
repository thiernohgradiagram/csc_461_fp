from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import librosa
import scipy

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


    # def extract_features_from_file(self, y, sr):
    #     """
    #     Extract features from an audio signal.
    #     """
    #     try:
    #         # Time-Domain Features
    #         rms = np.mean(librosa.feature.rms(y=y))
    #         zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    #         # Frequency-Domain Features
    #         spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    #         spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    #         spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    #         spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6), axis=1)
    #         spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    #         # Cepstral Features
    #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #         mfccs_mean = np.mean(mfccs, axis=1)
    #         delta_mfccs = np.mean(librosa.feature.delta(mfccs), axis=1)
    #         delta2_mfccs = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

    #         # Chroma Features
    #         chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    #         chroma_mean = np.mean(chroma, axis=1)

    #         # Tempo Features
    #         tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    #         # Tonnetz Features
    #         tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

    #         # Mel Spectrogram Features
    #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #         mel_spec_mean = np.mean(mel_spec, axis=1)

    #         # Combine all features into a single vector
    #         features = np.hstack([
    #             rms, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
    #             spectral_contrast, spectral_flatness, mfccs_mean, delta_mfccs, delta2_mfccs,
    #             chroma_mean, tempo, tonnetz, mel_spec_mean
    #         ])
    #         return features
    #     except Exception as e:
    #         print(f"Error extracting features: {e}")
    #         return None

    def extract_features_from_file(self, y, sr):
        """
        Extract features from an audio signal.
        """
        try:
            # Time-Domain Features
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))

            # Frequency-Domain Features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6), axis=1)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))


            # Harmonic and Percussive Energy
            harmonic = librosa.effects.harmonic(y)
            percussive = librosa.effects.percussive(y)
            harmonic_energy = np.mean(harmonic**2)
            percussive_energy = np.mean(percussive**2)

            # Spectral Skewness
            spectral_centroid_skewness = scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).flatten())

            # RMS of Harmonics
            rms_harmonic = np.mean(librosa.feature.rms(y=harmonic))

            # Silence Ratio
            silence_ratio = np.sum(librosa.feature.rms(y=y) < 0.01) / len(y)

            # Cepstral Features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            delta_mfccs = np.mean(librosa.feature.delta(mfccs), axis=1)
            delta2_mfccs = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

            # Tempo Features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Tonnetz Features
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

            # Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_mean = np.mean(mel_spec, axis=1)

            # Energy Band Ratios
            energy_low = np.mean(mel_spec[:40])   # First 40 Mel bands
            energy_mid = np.mean(mel_spec[40:80])  # Mid 40 Mel bands
            energy_high = np.mean(mel_spec[80:])  # Last 40 Mel bands

            # Combine all features into a single vector
            features = np.hstack([
                rms, zcr,
                spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast, spectral_flatness,
                harmonic_energy, percussive_energy,
                spectral_centroid_skewness,
                rms_harmonic,
                silence_ratio,
                mfccs_mean, delta_mfccs, delta2_mfccs,
                chroma_mean, tempo, tonnetz, mel_spec_mean, energy_low, energy_mid, energy_high
            ])
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None



    def process_file(self, file_info):
        file_path, genre_label, extractor = file_info
        try:
            y, sr = extractor.load_audio(file_path)
            features = extractor.extract_features_from_file(y, sr)
            if features is not None:
                return np.hstack([features, genre_label])
            else:
                print(f"Failed to extract features for {file_path}")
                return None
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


    def extract_features_all_files(self, dataset_path):
        genre_dict = {
            'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
            'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
        }

        # Gather all files with genre labels
        file_info_list = []
        for genre, label in genre_dict.items():
            genre_path = os.path.join(dataset_path, genre)
            if not os.path.isdir(genre_path):
                continue

            files = [os.path.join(genre_path, f) for f in os.listdir(genre_path) if f.endswith(".wav")]
            file_info_list.extend([(f, label, self) for f in files])

        # Process files in parallel
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_file, file_info_list))

        # Filter out None results and convert to numpy array
        data = np.array([r for r in results if r is not None])

        print(f"Feature extraction completed: {len(data)} files processed.")
        return data
    
    def getColumnNames(self):
        column_names = [
        # Time-Domain Features
        "RMS", "ZCR",

        # Frequency-Domain Features
        "Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff",
        "Spectral_Contrast_1", "Spectral_Contrast_2", "Spectral_Contrast_3",
        "Spectral_Contrast_4", "Spectral_Contrast_5", "Spectral_Contrast_6", "Spectral_Contrast_7",
        "Spectral_Flatness",

        # Harmonic and Percussive Energy
        "Harmonic_Energy", "Percussive_Energy",

        # Spectral Statistics
        "Spectral_Centroid_Skewness",

        # RMS of Harmonics
        "RMS_Harmonic",

        # Silence Ratio
        "Silence_Ratio",

        # Cepstral Features (MFCCs)
        "MFCC_1_Mean", "MFCC_2_Mean", "MFCC_3_Mean", "MFCC_4_Mean", "MFCC_5_Mean",
        "MFCC_6_Mean", "MFCC_7_Mean", "MFCC_8_Mean", "MFCC_9_Mean", "MFCC_10_Mean",
        "MFCC_11_Mean", "MFCC_12_Mean", "MFCC_13_Mean",
        "Delta_MFCC_1", "Delta_MFCC_2", "Delta_MFCC_3", "Delta_MFCC_4", "Delta_MFCC_5",
        "Delta_MFCC_6", "Delta_MFCC_7", "Delta_MFCC_8", "Delta_MFCC_9",
        "Delta_MFCC_10", "Delta_MFCC_11", "Delta_MFCC_12", "Delta_MFCC_13",
        "Delta2_MFCC_1", "Delta2_MFCC_2", "Delta2_MFCC_3", "Delta2_MFCC_4",
        "Delta2_MFCC_5", "Delta2_MFCC_6", "Delta2_MFCC_7", "Delta2_MFCC_8",
        "Delta2_MFCC_9", "Delta2_MFCC_10", "Delta2_MFCC_11", "Delta2_MFCC_12",
        "Delta2_MFCC_13",

        # Chroma Features
        "Chroma_1", "Chroma_2", "Chroma_3", "Chroma_4", "Chroma_5", "Chroma_6",
        "Chroma_7", "Chroma_8", "Chroma_9", "Chroma_10", "Chroma_11", "Chroma_12",

        # Tempo Features
        "Tempo",

        # Tonnetz Features
        "Tonnetz_1", "Tonnetz_2", "Tonnetz_3", "Tonnetz_4", "Tonnetz_5", "Tonnetz_6",

        # Mel Spectrogram Features
        ] + [f"Mel_Spec_{i+1}" for i in range(128)] + [
            # Energy Band Ratios
            "Energy_Low", "Energy_Mid", "Energy_High"
        ] + ["Genre"]

        return column_names
