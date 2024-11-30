import streamlit as st
import torch
from PIL import Image
import requests
import io
import sys
import os
import time

#To run this website, all you need to do is run " streamlit run website.py " in your terminal.
#This will then run a web server on your computer.


st.set_page_config(page_title="CSC 461", page_icon=":musical_note:", layout="wide")

#functions section
#this function will analysis of the selected file
def run_analysis(aFile):
    #found_genre = "this is a test genre"
    #add the parent directory to the current working directory so that we can access MLP class
    repository_root_directory = os.path.dirname(os.getcwd())
    rrd = "repository_root_directory:\t"


    if repository_root_directory not in sys.path:
        sys.path.append(repository_root_directory)

    #import the class
    from mlp import MLP

    #check to see if cuda is avalable
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("cuda status: ", device)


    #create a configuration dictionary. input size is the number of features in each sample
    config = {
        'input_size': 30,
        'output_size': 10,
        'hidden_layers': [256,64],
        'batch_size': 64,
        'n_epochs': 20,
        'learning_rate': 0.001
    }

    model = MLP(config['input_size'], config['output_size'], config['hidden_layers']).to(device)
    # Load the saved state dictionary
    savedModel = torch.load('model.pth')
    model.load_state_dict(torch.load('model.pth'))

    # Set the model to evaluation mode
    model.eval()

    #this will load a song and output the genre
    from data_preprocessor import DataPreprocessor
    from data_preprocessor_parallel_proc import DataPreprocessorParallelProc
    from pathlib import Path
    import librosa

    sample_rate = 22050                     # default sample rate of the dataset
    preprocessor = DataPreprocessor()
    audioProcessor = DataPreprocessorParallelProc()

    #get the path to the file
    #relative_path = 'song.mp3'
    relative_path = aFile

    # Get the absolute path
    file_path = os.path.abspath(relative_path)

    #print(f"The absolute path is: {file_path}")

    #print('Current working directory:', os.getcwd())

    # Preprocess the audio

    #file_path = aFile

    try:
                # Load the audio file with the correct sample rate
                y, sr = librosa.load(file_path, sr=sample_rate) if sample_rate != 22050 else librosa.load(file_path, sr=None)

                # Normalize the audio
                y = librosa.util.normalize(y)

                # Trim leading and trailing silence
                y, _ = librosa.effects.trim(y)

                
    except Exception as e:
        print(f"Error preprocessing file {file_path}: {e}")
    print("y is: ", y, "sr is: ", sr)


    #extract the features
    from features_extractor import FeaturesExtractor

    feature_extractor = FeaturesExtractor()
    data = feature_extractor.extract_features_from_file(y,sr)
    print("Data is: ",data)

    #convert the data to a tensor
    features = torch.tensor(data, dtype=torch.float32).unsqueeze(0) #make the data a tensor
    features = features.to(device) #make sure that it is using the same device. 

    #run the data through the model and get the predicted genre
    with torch.no_grad():  # No gradients needed for inference

        output = model(features)  # Forward pass
        predicted_class = torch.argmax(output, dim=1).item() 


    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco',4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',8: 'reggae', 9: 'rock'}  # Adjust based on your data
    predicted_genre = genre_mapping[predicted_class]
    print(f"Predicted Genre: {predicted_genre}")

    print("output is: ", output)
    print("predicted class is: ", predicted_class)
    print("that class is: ", predicted_genre)
    found_genre = predicted_genre
    return found_genre


def show_loading_animation():
    st.session_state["show_animation_container"] = True

analyzed_file = ""

#creating and initializing a shared values to be used between containers
#this will determine if the ai container should be shown
if "show_ai_container" not in st.session_state:
    st.session_state["show_ai_container"] = False

#this will hold the whole container whether it shows the animation or the result
if "show_animation_container" not in st.session_state:
    st.session_state["show_animation_container"] = False

#this will hold the data needed for the final analysis report
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = ""


#headder section
with st.container():
    st.title("CSC 461")
    st.write("---")

#this container contains the upload function
with st.container():
    st.subheader("choose a mp3 to analyze")
    # File uploader widget
    uploaded_file = st.file_uploader(label="Choose an MP3 file", type=["mp3"])
    # Check if a file is uploaded
    if uploaded_file is not None:
        #reset the ai container visibility if it was true
        st.session_state["show_ai_container"] = False
        st.write("File uploaded")
        
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        
        #show the continer that contains the analysis. This will first show the animation and when the analysis is done it will hide the animation and show the result
        show_loading_animation()
        analyzed_file = run_analysis(uploaded_file.name)
        #save the resulting text to the analysis_data session state
        st.session_state["analysis_data"] = analyzed_file

        st.session_state["show_animation_container"] = False
        #Show the container that contains the analsis result
        st.session_state["show_ai_container"] = True
        
    else:
        st.write("No file has been uploaded yet.")

#this is the analysis container.
with st.container():
    if st.session_state["show_animation_container"]:
        with st.container():
            st.write("---")
            st.subheader("Analyzing")
            st.image("https://media.giphy.com/media/3o7TKy6I2J9n3CWI9a/giphy.gif", width=300)
    #if not st.session_state["show_ai_container"] :
    #     with st.container():
    #        st.write("---")
    #        st.subheader("Analyzing")
    #        st.image("https://media.giphy.com/media/3o7TKy6I2J9n3CWI9a/giphy.gif", width=300)
    if st.session_state["show_ai_container"]:
        with st.container():
            st.write("---")
            st.subheader("Analysis")
            st.write(st.write(f"The resulting genre is: {st.session_state['analysis_data']}"))


