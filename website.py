import streamlit as st
import torch
import librosa
import os
import sys

#HOW TO RUN
#To run this website, all you need to do is run " streamlit run website.py " in your terminal. You must be in the same directory of the file
#This will then run a web server on your computer.


# Set the Streamlit configuration
st.set_page_config(page_title="CSC 461", page_icon=":musical_note:", layout="wide")

#------------------- Functon Section --------------------

#this processes the entire dataset and runs it through PCA. It then does the same to the input file and returns an array value.
def process_file(aFile):
    from features_extractor import FeaturesExtractor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np


    #try processing the audio file that was found
    try:
        y, sr = librosa.load(aFile, sr=22050)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
    
    except Exception as e:
        print(f"Error preprocessing file: {e}")

    #run the dataset through pca
    #get the dataset info
    gtzan_features_path = '_03_data_gtzan_features_labels.xlsx'      # Path to the GTZAN features file
    gtzan_features_file = pd.ExcelFile(gtzan_features_path)             # Load the Excel file   
    sheet_data = gtzan_features_file.parse('Sheet1')  

    # Extract the Mel_Spec_* features for PCA
    mel_spec_features = [col for col in sheet_data.columns if 'Mel_Spec_' in col]
    mel_spec_data = sheet_data[mel_spec_features]

    # Standardize the data before PCA
    scaler = StandardScaler()
    mel_spec_data_scaled = scaler.fit_transform(mel_spec_data)

    # Apply PCA and retain components that explain ~95% of the variance
    pca = PCA(n_components=0.98, random_state=42)
    mel_spec_pca = pca.fit_transform(mel_spec_data_scaled)

    # Create a DataFrame for the PCA-transformed features
    mel_spec_pca_df = pd.DataFrame(
        mel_spec_pca, columns=[f'PCA_Mel_Spec_{i+1}' for i in range(mel_spec_pca.shape[1])]
    )

    # Combine the PCA-transformed features with the rest of the dataset
    non_mel_spec_features = sheet_data.drop(columns=mel_spec_features)
    df_final = pd.concat([non_mel_spec_features, mel_spec_pca_df], axis=1)

    #run the user file through the same pca algorithm
    feature_extractor = FeaturesExtractor()
    data = feature_extractor.extract_features_from_file(y, sr)
    column_names = feature_extractor.getColumnNames()
    column_names = column_names[:-1]

    #reshape the data to (1,207)
    data = data.reshape(1, -1)
    #convert to a dataframe
    features_df = pd.DataFrame(data, columns=column_names)

    #Save to excel file
    features_df.to_excel('userFile.xlsx', index=False)

    gtzan_features_path_usr_file = 'userFile.xlsx'      # Path to the GTZAN features file
    gtzan_features_usr_file = pd.ExcelFile(gtzan_features_path_usr_file)             # Load the Excel file   
    sheet_data_usr_file = gtzan_features_usr_file.parse('Sheet1')  

    # Extract the Mel_Spec_* features for PCA
    mel_spec_features_usr_file = [col2 for col2 in sheet_data_usr_file.columns if 'Mel_Spec_' in col2]
    mel_spec_data_usr_file = sheet_data_usr_file[mel_spec_features_usr_file]

    # Standardize the mel_spec features using the scaler fitted on the large dataset
    mel_spec_usr_file_scaled = scaler.transform(mel_spec_data_usr_file)

    # Apply PCA (already fitted) to the scaled mel_spec features
    mel_spec_usr_file_pca_single = pca.transform(mel_spec_usr_file_scaled)
    # Create a DataFrame for the PCA-transformed features
    mel_spec_pca_df_usr_file = pd.DataFrame(
        mel_spec_usr_file_pca_single, columns=[f'PCA_Mel_Spec_{i+1}' for i in range(mel_spec_usr_file_pca_single.shape[1])]
    )

    # Combine the PCA-transformed features with the rest of the dataset
    non_mel_spec_features_usr_file = sheet_data_usr_file.drop(columns=mel_spec_features_usr_file)
    df_final_usr_file = pd.concat([non_mel_spec_features_usr_file, mel_spec_pca_df_usr_file], axis=1)
    
    
    user_file_data = df_final_usr_file.values
    return user_file_data
    #features = torch.tensor(user_file_data, dtype=torch.float32)#make the data a tensor.

def process_file_cnn(aFile):
    from features_extractor_cnn import FeaturesExtractor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np


    #try processing the audio file that was found
    try:
        y, sr = librosa.load(aFile, sr=22050)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
    
    except Exception as e:
        print(f"Error preprocessing file: {e}")

    target_shape = (150, 150)
    chunk_duration = 4
    overlap_duration = 2
    feature_extractor = FeaturesExtractor(target_shape, chunk_duration, overlap_duration)
    user_file_data, user_file_labels = feature_extractor.extract_features_from_file(aFile, y)
    return user_file_data, user_file_labels



#these functions will perform the analysis on the selected file


#this will run the mlp analysis
def run_analysis_MLP(aFile):
    import numpy as np

    #add the parent directory to the current working directory so that we can access MLP class
    #this will load the current working directory
    repository_root_directory = os.path.dirname(os.getcwd())
    rrd = "repository_root_directory:\t"
    print(rrd, repository_root_directory)

    if repository_root_directory not in sys.path:
        sys.path.append(repository_root_directory)
        print(rrd, "added to path")
    else:  
        print(rrd, "already in path")
    #add the parent directory to the current working directory so that we can access MLP class
    parent_dir = os.path.abspath(os.path.join(os.getcwd()))

    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        print(rrd, "added to path")
    else:  
        print(rrd, "already in path")

    from MLPNN import MLP

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #create a configuration dictionary.
    #72% test accuracy
    config = {
        'input_size': 146,
        'output_size': 10,
        'hidden_layers': [146, 146, 146],
        'batch_size': 25,
        'n_epochs': 100,
        'learning_rate': 0.002,
        'dropout_rate': 0.3,  # Add dropout to reduce overfitting
        'weight_decay': 0.0001  # Add L2 regularization
    }

    model = MLP(config['input_size'], config['output_size'], config['hidden_layers'], config['dropout_rate']).to(device)
    #load the saved state dictionary
    model.load_state_dict(torch.load('model.pth'))
    #set the model to evaluation mode
    model.eval()
    #run the data through pca and get a scaled value back
    user_file_data = process_file(aFile)
    features = torch.tensor(user_file_data, dtype=torch.float32)#make the data a tensor.
    features = features.to(device)
    

    #run the data through the model and get the predicted genre
    with torch.no_grad(): #No gradients needed for inference
        output = model(features)# Forward pass
        predicted_class = torch.argmax(output, dim=1).item()

    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class]

def run_analysis_NN(aFile):
    from NN import NeuralNetwork

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'input_size': 146,
        'hidden_size': 64,
        'num_classes': 10
    }

    model = NeuralNetwork(config['input_size'],config['hidden_size'],config['num_classes']).to(device)

    #load the nn model
    model.load_state_dict(torch.load('nn_model.pth'))

    model.eval()

     #since the model used a pca reduced dataset, reduce the input file in the same way
    user_file_data = process_file(aFile)
    features = torch.tensor(user_file_data, dtype=torch.float32)#make the data a tensor.
    features = features.to(device)

    with torch.no_grad(): #No gradients needed for inference
        output = model(features)# Forward pass
        predicted_class = torch.argmax(output, dim=1).item()

    print("predicted_class is: ", predicted_class)
    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class]

def run_analysis_LR(aFile):
    from joblib import load

    #get the trained model from the running of _05_mlr.ipynb
    optimized_logreg = load('lrModel.joblib')
    
    #since the model used a pca reduced dataset, reduce the input file in the same way
    user_file_data = process_file(aFile)
    predicted_class = optimized_logreg.predict(user_file_data)

    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class[0]]

def run_analysis_KNN(aFile):
    from joblib import load

    #get the trained model from the running of _05_mlr.ipynb
    optimized_knn = load('knn_model.joblib')
    
    #since the model used a pca reduced dataset, reduce the input file in the same way
    user_file_data = process_file(aFile)
    predicted_class = optimized_knn.predict(user_file_data)
    print("predicted_class is: ", predicted_class[0])
    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class[0]]


def run_analysis_CNN(aFile):
    import tensorflow as tf
    import numpy as np
    #load the model
    model = tf.keras.models.load_model('Trained_model.h5')

    #process the data
    user_file_data, user_file_label = process_file_cnn(aFile)
    #convert the data to an numpy array and transpose to the correct shape
    user_file_data_np =np.array(user_file_data)
    #user_file_data_np = np.reshape(user_file_data_np, (user_file_data_np.shape[0], 1, 150, 150))
    #run the data throught the model
    predictions = model.predict(user_file_data_np)
    predicted_class = np.argmax(predictions, axis=1)
    
    genre_mapping = {0: 'blues', 1: 'Classical', 2: 'Country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genre_mapping[predicted_class[0]]
    


#--------------------- Session state Setup -----------------------
# Initialize session states of all the data that will be tracked between containers
if "show_animation_container" not in st.session_state:
    st.session_state["show_animation_container"] = False
if "show_ai_container" not in st.session_state:
    st.session_state["show_ai_container"] = False
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = ""
if "accuracy_value" not in st.session_state:
    st.session_state["accuracy_value"] = "0"


#--------------------------- Header section ------------------------
st.title("CSC 461")
st.write("---")

#-------------------------- Main Website Section ------------------

# Upload container
with st.container():
    #Setup the radio buttons for the chosen AI Model to use 
    chosen_ai_model = st.radio("Choose a machine learning model to use", options=["MLP", "CNN", "KNN", "Logistic Regression", "NN"])

    print(chosen_ai_model)
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

        # Run analysis. Use the model that was chosen by the user
        if (chosen_ai_model == "MLP"):
            st.session_state["accuracy_value"] = "72%"
            analyzed_file = run_analysis_MLP(uploaded_file.name)
        elif (chosen_ai_model == "KNN"):
            analyzed_file = run_analysis_KNN(uploaded_file.name)
            st.session_state["accuracy_value"] = "52%"
        elif (chosen_ai_model == "Logistic Regression"):
            analyzed_file = run_analysis_LR(uploaded_file.name)
            st.session_state["accuracy_value"] = "67.5%"
        elif (chosen_ai_model == "NN"):
            analyzed_file = run_analysis_NN(uploaded_file.name)
            st.session_state["accuracy_value"] = "65%"
        elif (chosen_ai_model == "CNN"):
            analyzed_file = run_analysis_CNN(uploaded_file.name)
            st.session_state["accuracy_value"] = "90%"
        
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
        st.write(f"The accuracy of this model is: {st.session_state['accuracy_value']}")
