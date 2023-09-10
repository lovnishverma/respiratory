import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import tensorflow as tf
from keras.models import load_model

import librosa
import librosa.display
import seaborn as sns

#import sounddevice as sd
#from scipy.io.wavfile import write

#import pdb

st.title('Prediksi Penyakit Saluran Pernapasan')
with st.expander('**Konteks Model AI dan Database**'):
    #st.markdown('**Konteks Model AI dan Database:**')
    st.caption('*Made with ❤️  by Yose Marthin Giyay*')
    st.caption('Ini merupakan laman *interface* untuk model Convolutional Neural Network (CNN) yang dilatih menggunakan **TensorFlow 2.11.0**. Model ini dilatih dengan data dari **Respiratory Sound Database** yang dikemas 2 tim peneliti dengan subset populasi pasien di Portugal dan Yunani atas nama *International Conference on Biomedical Health Informatics* (ICHBI)')
    st.caption('Di sini _library_ **Librosa** digunakan untuk mengekstraksi MFCCs dari *file* audio. MFCC (*Mel-Frequency Cepstral Coefficients*) merupakan format representasi audio. Dengan proses matematis ini, fitur-fitur penting di jangkauan frekuensi alami telinga manusia dapat diekstraksi dari *file* audio dan dijadikan *input* ke model CNN untuk proses pelatihan model/prediksi.')
    st.caption('Database yang digunakan dapat dijelajah dan/atau diunduh di sini: https://bhichallenge.med.auth.gr/')
    st.caption('Jurnal ilmiah menyangkut pengumpulan data oleh tim dapat dilihat di sini: https://link.springer.com/chapter/10.1007/978-981-10-7419-6_6')
    st.caption('Project roadmap: developing a low-cost wireless stethoscope')
    
st.subheader('**Kategori diagnosis:**')
st.markdown('*- Sehat*   \n*- Bronkiektasis*   \n*- Bronkiolitis*   \n*- Penyakit Paru Obstruktif Kronis (PPOK)*   \n*- Pneumonia*   \n*- Infeksi Saluran Pernapasan Atas*')
st.subheader('Unggah *file* audio dan mulai prediksi')
st.caption('*Dalam pengembangan: rekam langsung di laman ini*. Idealnya audio yang digunakan direkam dengan stetoskop di area trakea, bisa gunakan mata stetoskop yang disambung dengan *mic* headset Bluetooth, misalnya. Untuk sekarang, bisa coba fitur *interface* dulu dengan rekaman pernapasan langsung dari mic HP.')
st.caption('**Silakan unggah *fail* audio .wav berdurasi ~20 detik**')

# Definisikan Function untuk prediksi
def predict_disease(model, features):
    # Predict
    prediction = model.predict(features)
    c_pred = np.argmax(prediction)
    
    return prediction, c_pred

# Muat model yang di latih - version mismatch, manual compile
model = load_model('./model/CNN-MFCC.h5', compile=False)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Label
clabels = ['Bronchiectasis', 'Bronchiolitis', 'Chronic Obstructive Pulmonary Disease (COPD)', 'Healthy', 'Pneumonia', 'Upper Respiratory Tract Infection (URTI)']
clabels_idn = ['Bronkiektasis', 'Bronkiolitis', 'Penyakit Paru Obstruktif Kronis (PPOK)', 'Sehat', 'Radang Paru-Paru', 'Infeksi Saluran Pernapasan Atas']

# Create a form for input and output components
with st.form(key="prediction_form"):
    # Upload and display audio file
    uploaded_file = st.file_uploader("Pilih *file* audio (hanya format .WAV)")
    
    # Proses audio yang diunggah, ekstraksi MFCCs
    if uploaded_file is not None:
        # Memuat berkas audio
        audio, sample_rate = librosa.load(uploaded_file, duration=20)
        
        # Tampilkan Spektogram Mel
        st.markdown('Mel Spectrogram')
        fig, ax = plt.subplots()
        sns.heatmap(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sample_rate), ref=np.max))
        st.pyplot(fig)

        # Ekstraksi MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Padding dimensi, dari (20, 862) ke (1, 40, 862, 1)
        max_pad_len = 862
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        features = np.expand_dims(np.array(mfccs), axis=(0, -1))

    # Submit the prediction request
    submit_button = st.form_submit_button("Prediksi kemungkinan penyakit")

    # Display the prediction results
    if submit_button:
        prediction, c_pred = predict_disease(model, features)
        max_value = np.max(prediction)
        formatted_max = np.format_float_positional(max_value*100, precision=2)
        st.title('Prediksi: ')
        st.subheader(f'**{clabels_idn[c_pred]}**: {formatted_max}%')
        st.subheader(f'*{clabels[c_pred]}*')
