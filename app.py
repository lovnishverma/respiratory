import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import tensorflow as tf
from keras.models import load_model

import librosa
import librosa.display
import seaborn as sns

# Import sounddevice as sd
# from scipy.io.wavfile import write

# Import pdb

st.title('Respiratory Tract Disease Prediction')
with st.expander('**AI Model and Database Context**'):
    st.caption('*Made with ❤️*')
    st.caption('This is an interface for a Convolutional Neural Network (CNN) model trained using **TensorFlow 2.11.0**. The model is trained on data from the **Respiratory Sound Database**, collected by two research teams with patient subsets in Portugal and Greece, presented at the International Conference on Biomedical Health Informatics (ICHBI).')
    st.caption('Librosa library is used here to extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio files. MFCCs represent the audio in a mathematical format, allowing extraction of important features in the natural frequency range of the human ear from audio files for input into the CNN model during training/prediction.')
    st.caption('The used database can be explored and/or downloaded here: https://bhichallenge.med.auth.gr/')
    st.caption('Scientific papers related to data collection by the research teams can be found here: https://link.springer.com/chapter/10.1007/978-981-10-7419-6_6')
    st.caption('Project roadmap: developing a low-cost wireless stethoscope')

st.subheader('**Diagnosis Categories:**')
st.markdown('*- Healthy*   \n*- Bronchiectasis*   \n*- Bronchiolitis*   \n*- Chronic Obstructive Pulmonary Disease (COPD)*   \n*- Pneumonia*   \n*- Upper Respiratory Tract Infection (URTI)*')
st.subheader('Upload an audio file and start prediction')
st.caption('*In development: record directly on this page*. Ideally, use audio recorded with a stethoscope in the tracheal area. You can use a Bluetooth headset with a stethoscope attachment, for example. For now, try the interface feature first with direct respiratory recordings from the phone microphone.')
st.caption('**Please upload a .wav audio file with a duration of ~20 seconds**')

# Define Function for prediction
def predict_disease(model, features):
    # Predict
    prediction = model.predict(features)
    c_pred = np.argmax(prediction)
    
    return prediction, c_pred

# Load the trained model - version mismatch, manual compile
model = load_model('./model/CNN-MFCC.h5', compile=False)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Labels
clabels = ['Bronchiectasis', 'Bronchiolitis', 'Chronic Obstructive Pulmonary Disease (COPD)', 'Healthy', 'Pneumonia', 'Upper Respiratory Tract Infection (URTI)']
clabels_idn = ['Bronchiectasis', 'Bronchiolitis', 'Chronic Obstructive Pulmonary Disease (COPD)', 'Healthy', 'Pneumonia', 'Upper Respiratory Tract Infection (URTI)']

# Create a form for input and output components
with st.form(key="prediction_form"):
    # Upload and display audio file
    uploaded_file = st.file_uploader("Choose an audio file (only .WAV format)")
    
    # Process the uploaded audio, extract MFCCs
    if uploaded_file is not None:
        # Load the audio file
        audio, sample_rate = librosa.load(uploaded_file, duration=20)
        
        # Display Mel Spectrogram
        st.markdown('Mel Spectrogram')
        fig, ax = plt.subplots()
        sns.heatmap(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sample_rate), ref=np.max))
        st.pyplot(fig)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad dimensions, from (20, 862) to (1, 40, 862, 1)
        max_pad_len = 862
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        features = np.expand_dims(np.array(mfccs), axis=(0, -1))

    # Submit the prediction request
    submit_button = st.form_submit_button("Predict the likelihood of disease")

    # Display the prediction results
    if submit_button:
        prediction, c_pred = predict_disease(model, features)
        max_value = np.max(prediction)
        formatted_max = np.format_float_positional(max_value*100, precision=2)
        st.title('Prediction: ')
        st.subheader(f'**{clabels[c_pred]}**: {formatted_max}%')
        st.subheader(f'*{clabels_idn[c_pred]}*')
