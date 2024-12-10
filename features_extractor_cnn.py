from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import librosa
import tensorflow as tf

class FeaturesExtractor:

    def __init__(self, target_shape, chunk_duration, overlap_duration):
        """
        Initialize the FeatureExtractor.

        Parameters:
        - target_shape (tuple): The target shape for resized mel spectrograms.
        - chunk_duration (int): Duration of each audio chunk in seconds.
        - overlap_duration (int): Overlap duration between consecutive chunks in seconds.
        """
        self.target_shape = target_shape
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

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

    def extract_features_from_file(self, file_path, label):
        """
        Extract features from a single audio file.
        Returns features and labels for this file.
        """
        data = []
        labels = []

        try:
            # Load the audio file
            y, sample_rate = self.load_audio(file_path)

            # Convert durations to samples
            chunk_samples = self.chunk_duration * sample_rate
            overlap_samples = self.overlap_duration * sample_rate

            # Calculate the number of chunks
            num_chunks = int(np.ceil((len(y) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

            # Iterate over each chunk
            for i in range(num_chunks):
                start = i * (chunk_samples - overlap_samples)
                end = start + chunk_samples

                # Extract chunk
                chunk = y[start:end]

                 # Check if the chunk is long enough to process
                if len(chunk) < chunk_samples:
                    print(f"Ignoring incomplete chunk from file: {file_path}")
                    continue

                # Generate Mel spectrogram
                mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), self.target_shape)

                # Append results
                data.append(mel_spectrogram)
                labels.append(label)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        return data, labels
    
    def process_file_helper(self, file_info):
        """
        Helper method to process a single file. This method is necessary for parallel processing.
        """
        file_path, label = file_info
        return self.extract_features_from_file(file_path, label)


    def extract_features_all_files(self, dataset_path):
        """
        Extract features from all files in the dataset using parallel processing.
        """
        genre_dict = {
            'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
            'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
        }

        # Collect all file paths and labels
        file_info_list = []
        for genre, label in genre_dict.items():
            genre_path = os.path.join(dataset_path, genre)
            if not os.path.isdir(genre_path):
                raise ValueError(f"Directory not found: {genre_path}")

            for filename in os.listdir(genre_path):

                if filename.endswith('.wav'):
                    file_info_list.append((os.path.join(genre_path, filename), label))

        # Calculate available CPUs minus 1
        available_cpus = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
        max_workers = max(1, available_cpus - 1)  # Ensure at least 1 worker is used
        
        print(f"Using {max_workers} CPUs for parallel processing.")

        # Parallel processing
        all_data = []
        all_labels = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use the helper method
            results = executor.map(self.process_file_helper, file_info_list)

        for data, labels in results:
            all_data.extend(data)
            all_labels.extend(labels)

        print(f"Feature extraction completed: {len(all_data)} chunks processed.")
        return np.array(all_data), np.array(all_labels)