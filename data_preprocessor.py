import os
import librosa
import soundfile as sf

class DataPreprocessor:

    def __init__(self):
        """
        Initializes the DataPreProcessor class.
        """
        pass

    def preprocess_audio(self, file_path, sample_rate):
        """
        Preprocess an audio file by normalizing the audio and trimming silence.

        Parameters:
        - file_path (str): Path to the audio file.

        Returns:
        - y (np.ndarray): Preprocessed audio time series.
        - sr (int): Sampling rate of the processed audio.
        """
        try:
            # Load the audio file with the correct sample rate, only if the sample rate is different from the default
            y, sr = librosa.load(file_path, sr=sample_rate) if sample_rate != 22050 else librosa.load(file_path, sr=None)

            # Normalize the audio
            y = librosa.util.normalize(y)

            # Trim leading and trailing silence
            y, _ = librosa.effects.trim(y)

            return y, sr
        except Exception as e:
            print(f"Error preprocessing file {file_path}: {e}")
            return None, None

    def save_preprocessed_audio(self, genre, file_name, output_path, y, sr):
        """
        Save the preprocessed audio to the output path if not already saved.

        Parameters:
        - genre (str): Genre of the audio file.
        - file_name (str): Name of the audio file.
        - y (np.ndarray): Preprocessed audio time series.
        - sr (int): Sampling rate of the audio file.
        """
        genre_path = os.path.join(output_path, genre)
        os.makedirs(genre_path, exist_ok=True)

        output_file = os.path.join(genre_path, file_name.replace(".au", ".wav"))

        # Check if the file already exists
        if not os.path.exists(output_file):
            sf.write(output_file, y, sr, format='WAV')
            print(f"Saved: {output_file}")
        else:
            print(f"File already exists: {output_file}")

    def process_dataset(self, dataset_path, output_path, sample_rate):
        """
        Processes the entire dataset, preprocesses the audio, and saves it.
        """
        genres = os.listdir(dataset_path)  # Assuming each genre is a folder

        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith(".au"):  # Process only .au files
                        file_path = os.path.join(genre_path, file)

                        # Preprocess the audio
                        y, sr = self.preprocess_audio(file_path, sample_rate)

                        if y is not None:
                            # Save the preprocessed audio
                            self.save_preprocessed_audio(genre, file, output_path, y, sr)
