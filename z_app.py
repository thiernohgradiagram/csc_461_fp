import streamlit as st
import numpy as np
import tensorflow as tf
import os
import tempfile
from features_extractor_cnn import FeaturesExtractor
import librosa
import librosa.display
import matplotlib.pyplot as plt

#Plotting Melspectrogram of Entire Audio
def plot_melspectrogram(y, sr):
    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert to decibels (log scale)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel-Spectrogram")
    plt.tight_layout()

    # Return the figure object
    return fig

# Set page title and layout
st.set_page_config(
    page_title="GTZAN Genre Classification App",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="auto"
)

# Set page title
st.title("GTZAN Genre Classification App")

# Add some instructions
st.write("Upload an audio file (WAV or MP3) and this app will predict its genre.")

# Mapping from label indices to genre names
CLASS_LABELS = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Load the trained model
rrd = os.path.dirname(os.getcwd())
model_path = os.path.join(rrd, "this_studio", "csc_461_fp", "models", "Trained_model.h5") 
model = tf.keras.models.load_model(model_path)

# Initialize the FeaturesExtractor
# You can add a sidebar for parameter adjustments
st.sidebar.title("Feature Extraction Settings")
chunk_duration = st.sidebar.slider("Chunk Duration (seconds)", 1, 10, 4)
overlap_duration = st.sidebar.slider("Overlap Duration (seconds)", 0, 5, 2)

feature_extractor = FeaturesExtractor(
    target_shape=(150, 150),
    chunk_duration=chunk_duration,
    overlap_duration=overlap_duration
)

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Display the uploaded audio
    st.audio(uploaded_file, format='audio/wav')

    # ---- Visualizing the waveform of the uploaded audio ----
    try:
        y, sr = librosa.load(temp_file_path, sr=None)
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set_title("Waveform of Uploaded Audio")
        st.pyplot(fig_wave)

        # Plot the entire mel-spectrogram
        fig_mel_spec = plot_melspectrogram(y, sr)
        st.pyplot(fig_mel_spec)
    except:
        st.write("Could not display waveform.")

    # Extract features
    data, _ = feature_extractor.extract_features_from_file(temp_file_path, label=0)

    if len(data) == 0:
        st.write("No valid audio chunks were extracted. Please try another file.")
    else:
        # Convert data to a NumPy array for prediction
        data_np = np.stack([chunk.numpy() for chunk in data], axis=0)

        # Make predictions
        predictions = model.predict(data_np)
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class_idx = np.argmax(avg_prediction)
        predicted_genre = CLASS_LABELS[predicted_class_idx]

        # Show the main prediction
        st.write("**Predicted Genre:**", predicted_genre)

        # ---- Display Top-3 Predictions ----
        sorted_indices = np.argsort(avg_prediction)[::-1]  # descending order
        top_3_indices = sorted_indices[:3]
        st.write("## Top Predictions")
        for i, idx in enumerate(top_3_indices, start=1):
            st.write(f"{i}. {CLASS_LABELS[idx]} - {avg_prediction[idx]*100:.2f}%")

        # ---- Add a confidence progress bar for the top prediction ----
        predicted_prob = avg_prediction[predicted_class_idx]
        st.progress(int(predicted_prob * 100))
        st.write(f"Model Confidence: {predicted_prob*100:.2f}%")

        # ---- Maintain a Prediction History in Session State ----
        if 'prediction_counts' not in st.session_state:
            st.session_state.prediction_counts = {genre: 0 for genre in CLASS_LABELS}
        st.session_state.prediction_counts[predicted_genre] += 1

        st.write("## Prediction History")
        for genre, count in st.session_state.prediction_counts.items():
            st.write(f"{genre}: {count}")

        # ---- Downloadable Report (Text) ----
        report = f"Predicted Genre: {predicted_genre}\n"
        for i, idx in enumerate(top_3_indices, start=1):
            report += f"{i}. {CLASS_LABELS[idx]} - {avg_prediction[idx]*100:.2f}%\n"
        st.download_button("Download Prediction Report", report, file_name="prediction_report.txt")

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# ---- Add a section about the model/dataset ----
st.write("## About the Model")
st.write("""
This model was trained on the GTZAN dataset, which consists of 10 genres:
Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

The model is a Convolutional Neural Network (CNN) trained on Mel spectrogram representations of audio clips.
It can classify short audio segments into one of these 10 genres.
""")

