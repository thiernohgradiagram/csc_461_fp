import streamlit as st
import torch
import librosa

#HOW TO RUN
#To run this website, all you need to do is run " streamlit run website.py " in your terminal. You must be in the same directory of the file
#This will then run a web server on your computer.


# Set the Streamlit configuration
st.set_page_config(page_title="CSC 461", page_icon=":musical_note:", layout="wide")

#------------------- Functon Section --------------------
#this function will perform the analysis on the selected file

def run_analysis(aFile):
    from mlp import MLP
    from features_extractor import FeaturesExtractor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #create a configuration dictionary.
    config = {
        'input_size': 30,
        'output_size': 10,
        'hidden_layers': [256, 64],
        'batch_size': 64,
        'n_epochs': 20,
        'learning_rate': 0.001,
    }

    model = MLP(config['input_size'], config['output_size'], config['hidden_layers']).to(device)
    #load the saved state dictionary
    model.load_state_dict(torch.load('model.pth'))
    #set the model to evaluation mode
    model.eval()
    #try processing the audio file that was found
    try:
        y, sr = librosa.load(aFile, sr=22050)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
    
    except Exception as e:
        print(f"Error preprocessing file: {e}")
    
    feature_extractor = FeaturesExtractor()
    data = feature_extractor.extract_features_from_file(y, sr)
    features = torch.tensor(data, dtype=torch.float32).unsqueeze(0)#make the data a tensor
    features = features.to(device)#make sure it is using the same device. 

    #run the data through the model and get the predicted genre
    with torch.no_grad(): #No gradients needed for inference
        output = model(features)# Forward pass
        predicted_class = torch.argmax(output, dim=1).item()

    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class]

#--------------------- Session state Setup -----------------------
# Initialize session states of all the data that will be tracked between containers
if "show_animation_container" not in st.session_state:
    st.session_state["show_animation_container"] = False
if "show_ai_container" not in st.session_state:
    st.session_state["show_ai_container"] = False
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = ""

#--------------------------- Header section ------------------------
st.title("CSC 461")
st.write("---")

# Upload container
with st.container():
    st.subheader("Choose an MP3 to Analyze")
    #setup the file uploader. Only allow mp3's to be uploaded
    uploaded_file = st.file_uploader(label="Choose an MP3 file", type=["mp3"])
    #check if a file is uploaded
    if uploaded_file is not None:
        # Reset states if you are doing another rerun of the website. The ai container will be invisible and the animation container will be visible
        st.session_state["show_ai_container"] = False
        st.session_state["show_animation_container"] = True

        # Save file locally
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        # Create placeholder for animation
        animation_placeholder = st.empty()

        # Display loading animation
        with animation_placeholder.container():
            st.subheader("Analyzing")
            st.image("LoadAnimation.gif", width=300)

        # Run analysis
        analyzed_file = run_analysis(uploaded_file.name)
        #save the analysis to the analyzed file variable
        st.session_state["analysis_data"] = analyzed_file

        # Update state to hide animation and show results
        st.session_state["show_animation_container"] = False
        st.session_state["show_ai_container"] = True
        animation_placeholder.empty()  # Clear the animation container
    else:
        st.write("No file has been uploaded yet.")
        #hide the ai continer if you clear out the uploaded file
        st.session_state["show_ai_container"] = False
        

# Analysis container
if st.session_state["show_ai_container"]:
    with st.container():
        st.write("---")
        st.subheader("Analysis")
        st.write(f"The resulting genre is: {st.session_state['analysis_data']}")
