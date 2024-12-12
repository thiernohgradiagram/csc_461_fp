import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import torch
import torch.nn as nn
import os

from features_extractor_cnn import FeaturesExtractor

#####################################################################################
class CNNModel(nn.Module):
    def __init__(self, input_shape=(150, 150, 1), num_classes=10):
        super(CNNModel, self).__init__()
        in_channels = input_shape[2]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # After dimension calculations for input 150x150:
        # Final feature map size after all pooling and convs is (512, 2, 2)
        # Flattened size = 512 * 2 * 2 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1200),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(1200, num_classes)
        )

    def forward(self, x):
        # x: (N, H, W, C) -> permute to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = self.classifier(x)
        return x

#############################################################################################

# Set up the page
st.set_page_config(
    page_title="GTZAN Genre Classification App",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("GTZAN Genre Classification App")

CLASS_LABELS = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# Load PyTorch model
rrd = os.path.dirname(os.getcwd())
model_path = os.path.join(rrd, "this_studio", "csc_461_fp", "notebooks", "cnn_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the same architecture
input_shape = (150, 150, 1)  # Your input shape
model = CNNModel(input_shape=input_shape, num_classes=10)

# Load the state dictionary
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

st.sidebar.title("Feature Extraction Settings")
chunk_duration = st.sidebar.slider("Chunk Duration (seconds)", 1, 10, 4)
overlap_duration = st.sidebar.slider("Overlap Duration (seconds)", 0, 5, 2)

feature_extractor = FeaturesExtractor(
    target_shape=(150, 150),
    chunk_duration=chunk_duration,
    overlap_duration=overlap_duration
)

# Function to plot mel spectrogram of entire audio
def plot_melspectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel-Spectrogram")
    plt.tight_layout()
    return fig

# Function to make predictions with PyTorch model
def predict_genre(data_np):
    with torch.no_grad():
        # Convert numpy array to PyTorch tensor
        data_tensor = torch.FloatTensor(data_np).to(device)
        # Make predictions
        outputs = model(data_tensor)
        # Convert to probabilities
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        # Move back to CPU and convert to numpy for averaging
        predictions_np = predictions.cpu().numpy()
        # Average predictions across chunks
        avg_prediction = np.mean(predictions_np, axis=0)
        predicted_class_idx = np.argmax(avg_prediction)
        return predicted_class_idx, avg_prediction

# --- Upload Section ---
st.subheader("Upload an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    st.audio(uploaded_file, format='audio/wav')

    # Load and display uploaded audio waveform
    y, sr = librosa.load(temp_file_path, sr=None)
    fig_wave, ax_wave = plt.subplots(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title("Waveform of Uploaded Audio")
    st.pyplot(fig_wave)

    # Plot mel-spectrogram
    fig_mel_spec = plot_melspectrogram(y, sr)
    st.pyplot(fig_mel_spec)

    # Extract features and predict
    data, _ = feature_extractor.extract_features_from_file(temp_file_path, label=0)

    if len(data) == 0:
        st.write("No valid audio chunks were extracted. Please try another file.")
    else:
        data_np = np.stack([chunk.numpy() for chunk in data], axis=0)
        predicted_class_idx, avg_prediction = predict_genre(data_np)
        predicted_genre = CLASS_LABELS[predicted_class_idx]

        st.write("**Predicted Genre (Uploaded Audio):**", predicted_genre)
        
        # Display prediction confidence
        st.write("Prediction Confidence:")
        for i, (genre, conf) in enumerate(zip(CLASS_LABELS, avg_prediction)):
            st.write(f"{genre}: {conf*100:.2f}%")

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

###############################################################################################

# # --- Recording Section ---
# st.subheader("Record an Audio Clip")
# st.write("Press the button below to start and stop the recording.")

# audio_data = st_audiorec()

# if audio_data is not None:
#     # audio_data is WAV file data in bytes
#     temp_rec_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     with open(temp_rec_file.name, "wb") as f:
#         f.write(audio_data)

#     recorded_file_path = temp_rec_file.name

#     # Display recorded audio
#     st.audio(recorded_file_path, format='audio/wav')

#     # Load and process recorded audio
#     y_rec, sr_rec = librosa.load(recorded_file_path, sr=None)
#     fig_wave_rec, ax_wave_rec = plt.subplots(figsize=(10,3))
#     librosa.display.waveshow(y_rec, sr=sr_rec, ax=ax_wave_rec)
#     ax_wave_rec.set_title("Waveform of Recorded Audio")
#     st.pyplot(fig_wave_rec)

#     # Plot mel-spectrogram for recorded audio
#     fig_mel_spec_rec = plot_melspectrogram(y_rec, sr_rec)
#     st.pyplot(fig_mel_spec_rec)

#     # Extract features and predict
#     data_rec, _ = feature_extractor.extract_features_from_file(recorded_file_path, label=0)

#     if len(data_rec) == 0:
#         st.write("No valid audio chunks were extracted from the recording.")
#     else:
#         data_rec_np = np.stack([chunk.numpy() for chunk in data_rec], axis=0)
#         predicted_class_idx_rec, avg_prediction_rec = predict_genre(data_rec_np)
#         predicted_genre_rec = CLASS_LABELS[predicted_class_idx_rec]

#         st.write("**Predicted Genre (Recorded Audio):**", predicted_genre_rec)

#         # Display prediction confidence for recorded audio
#         st.write("Prediction Confidence:")
#         for i, (genre, conf) in enumerate(zip(CLASS_LABELS, avg_prediction_rec)):
#             st.write(f"{genre}: {conf*100:.2f}%")

#     if os.path.exists(recorded_file_path):
#         os.remove(recorded_file_path)


# # About the Model
# st.write("## About the Model")
# st.write("""
# This model was trained on the GTZAN dataset, which consists of 10 genres:
# Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

# The model is a CNN trained on Mel spectrogram representations of audio clips.
# It can classify short audio segments into one of these 10 genres.
# """)

# --- Recording Section ---
st.subheader("Record an Audio Clip")
st.write("Press the button below to start and stop the recording.")

audio_data = st_audiorec()

if audio_data is not None:
    # Show a spinner while we process the recorded audio
    with st.spinner("Processing recorded audio..."):
        # audio_data is WAV file data in bytes
        temp_rec_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_rec_file.name, "wb") as f:
            f.write(audio_data)

        recorded_file_path = temp_rec_file.name

        # Display recorded audio
        # st.audio(recorded_file_path, format='audio/wav')

        # Load and process recorded audio
        y_rec, sr_rec = librosa.load(recorded_file_path, sr=None)
        fig_wave_rec, ax_wave_rec = plt.subplots(figsize=(10,3))
        librosa.display.waveshow(y_rec, sr=sr_rec, ax=ax_wave_rec)
        ax_wave_rec.set_title("Waveform of Recorded Audio")
        st.pyplot(fig_wave_rec)

        # Plot mel-spectrogram for recorded audio
        fig_mel_spec_rec = plot_melspectrogram(y_rec, sr_rec)
        st.pyplot(fig_mel_spec_rec)

        # Extract features and predict
        data_rec, _ = feature_extractor.extract_features_from_file(recorded_file_path, label=0)

        if len(data_rec) == 0:
            st.write("No valid audio chunks were extracted from the recording.")
        else:
            data_rec_np = np.stack([chunk.numpy() for chunk in data_rec], axis=0)
            predicted_class_idx_rec, avg_prediction_rec = predict_genre(data_rec_np)
            predicted_genre_rec = CLASS_LABELS[predicted_class_idx_rec]

            st.write("**Predicted Genre (Recorded Audio):**", predicted_genre_rec)

            # Display prediction confidence for recorded audio
            st.write("Prediction Confidence:")
            for i, (genre, conf) in enumerate(zip(CLASS_LABELS, avg_prediction_rec)):
                st.write(f"{genre}: {conf*100:.2f}%")

        if os.path.exists(recorded_file_path):
            os.remove(recorded_file_path)

# About the Model
st.write("## About the Model")
st.write("""
This model was trained on the GTZAN dataset, which consists of 10 genres:
Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

The model is a CNN trained on Mel spectrogram representations of audio clips.
It can classify short audio segments into one of these 10 genres.
""")
