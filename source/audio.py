# -*- coding: utf-8 -*-
"""
@file    : CSV Converter
@brief   : Handles TXT to CSV file conversion.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import io
from io import BytesIO
import csv
import numpy as np
import pandas as pd
import streamlit as st
import pydub
import math
import librosa.display
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io.wavfile as wavfile

def wav_plot():

    st.markdown("# WAV Plot 📢")
    st.sidebar.markdown("# WAV Plot 📢")

    uploaded_files = st.file_uploader(
        "Elige los archivos WAV para graficar",
        type=["wav"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos WAV para graficar",
    )

    if len(uploaded_files) > 0:
        threshold = st.number_input("Sensitivity Threshold", value=0.18, placeholder="Sensitivity Threshold")
        for uploaded_file in uploaded_files:
            #data = uploaded_file.read()
            #bytes_data = uploaded_file.getvalue()

            name = uploaded_file.name
            format = name[-3:]
            st.audio(uploaded_file, format="audio/wav", start_time=0)

            #if st.checkbox(f'Graficar {name}'):
            filename = f'{name[:-4]}'
            x, y, sr = get_data(uploaded_file)
            d, t, p = detect_discontinuities(x, y, sr, threshold)

            plot_waveform(x, y, t, p, name, sr)

            #y = handle_uploaded_audio_file(uploaded_file)
            #st.write(f'Frecuencia de muestreo: {sr}')
            #df = pd.DataFrame(y)
            #st.line_chart(df)
            #st.write(df)
            #

        return True

    return False


def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_wav(uploaded_file)
    #st.write(a.sample_width)
    a = pydub.AudioSegment(
        # raw audio data (bytes)
        data=a,

        # 2 byte (16 bit) samples
        sample_width=2,

        # 44.1 kHz frame rate
        frame_rate=10000,

        # stereo
        channels=2
    )

    samples = a.get_array_of_samples()
    #fp_arr = np.array(samples).T.astype(np.float32)
    #fp_arr /= np.iinfo(samples.typecode).max
    #st.write(fp_arr.shape)
    return samples #, fp_arr.shape[0]


def plot_waveform(x, y, d, p, filename, sr):
    # Streamlit
    #st.line_chart(y)

    # Plotly
    #fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=x, y=y, name=f'Waveform', line={'width': 2}, mode='lines'))
    fig.add_trace(go.Scatter(x=d, y=p, name=f'Discontinuities', line={'width': 2}, mode='markers', marker=dict(
                color='LightGreen',
                size=8,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),))
    fig.update_layout(
        title=f"{filename}",
        xaxis_title="Time [s]",
        yaxis_title="Amplitude"
    )

    fig.add_annotation(
            dict(
                font=dict(color='black',size=12),
                x=1.03,
                y=0.07,
                showarrow=False,
                text=f'SF: {sr}[Hz]',
                textangle=0,
                xanchor='left',
                yanchor='top',
                xref="paper",
                yref="paper"
            )
        )

    # Displaying the chart
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # MatplotLib
    #fig, ax = plt.subplots(figsize=(10, 4))
    #librosa.display.waveshow(y, sr=sr)
    #plt.title('Waveform')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Amplitude')
    #st.pyplot(fig)




# Function to plot waveform
def get_data(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    x = np.linspace(0, duration, len(y))
    return x, y, sr

# Function to detect discontinuities in WAV signal
def detect_discontinuities(time, data, sample_rate, threshold=0.1):

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Calculate differences between consecutive samples
    differences = np.abs(np.diff(data))

    # Find indices where differences exceed the threshold
    discontinuity_indices = np.where(differences > threshold)[0] + 1

    p = []
    for i in discontinuity_indices:
        p.append(data[i])

    t = []
    for i in discontinuity_indices:
        t.append(time[i])

    return discontinuity_indices, t, p
