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

        for uploaded_file in uploaded_files:
            data = uploaded_file.read()
            bytes_data = uploaded_file.getvalue()

            name = uploaded_file.name
            format = name[-3:]
            st.audio(uploaded_file, format="audio/wav", start_time=0)

            if st.checkbox(f'Graficar {name}'):
                y = handle_uploaded_audio_file(uploaded_file)
                filename = f'{name[:-4]}'
                df = pd.DataFrame(y)
                #st.line_chart(df)
                st.write(df)
                #st.write(f'Frecuencia de muestreo: {sr}')

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
    return samples#, fp_arr.shape[0]
