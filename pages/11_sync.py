# -*- coding: utf-8 -*-
"""
@file    : 10_sync.py
@brief   : Handles synchronization of data.
@date    : 2026/03/24
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import streamlit as st
import pandas as pd
import openpyxl
import heartpy as hp
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.stats import norm
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
from moviepy import VideoFileClip
import tempfile
from scipy.io import wavfile
from scipy.signal import resample



im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)


def butter_lowpass_filter(data, cutoff, fs, order):
    if order == 0:
        return data
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalise frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)  # Filter data
    return y


def plot_data(mag_abs, mag_filt, epoch, start_epoch, end_epoch):

    plot_df = pd.DataFrame({
        "epoch": epoch,
        "abs_mag": mag_abs,
        "filtered": mag_filt
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["epoch"],
            y=plot_df["abs_mag"],
            mode="lines",
            name="abs_mag"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["epoch"],
            y=plot_df["filtered"],
            mode="lines",
            name="filtered"
        )
    )

    # Vertical lines
    fig.add_vline(x=start_epoch, line_dash="dash", line_color="green")
    fig.add_vline(x=end_epoch, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Signal Comparison",
        xaxis_title="Epoch",
        yaxis_title="Magnitude",
        legend_title="Signals"
    )

    st.plotly_chart(fig, use_container_width=True)



def fix_data(df):
    df["epoch"] = df["timestamp_s"].str.replace(",", ".").astype(float)
    df["time"] = pd.to_datetime(df["epoch"], unit="s")
    return df


def get_magnitudes(df):
    x = df["x_mG"].values
    y = df["y_mG"].values
    z = df["z_mG"].values

    mag = np.sqrt(x**2 + y**2 + z**2)
    df["magnitude"] = mag
    mag_abs = np.abs(mag - np.mean(mag))
    df["magnitude_abs"] = mag_abs
    return df


def get_sf(df):
    dt = np.diff(df["epoch"])
    mean_dt = np.mean(dt)
    fs = 1/mean_dt
    return fs


def sync_data():
    # Streamlit app setup
    st.title("Análisis Tobii 👓")
    st.sidebar.markdown("# Análisis Tobii 👓")

    st.write("1. Cargar datos procesados del algoritmo:")
    # Cargar archivos CSV
    acc_file = st.file_uploader("Cargar archivo CSV con datos de aceleración", type=["csv"])
    ecg_file = st.file_uploader("Cargar archivo CSV con datos de electrocardiograma", type=["csv"])
    video_file = st.file_uploader("Cargar archivo de video", type=["mp4"])



    if acc_file and ecg_file and video_file:
        acc_data = pd.read_csv(acc_file, sep=";")
        ecg_data = pd.read_csv(ecg_file, sep=";")
        
        ecg_data = fix_data(ecg_data)
        acc_data = fix_data(acc_data)

        #st.write(acc_data)

        acc_data = get_magnitudes(acc_data)
        #st.write(acc_data)

        fs = get_sf(acc_data)
        st.write(f"Frecuencia de muestreo estimada: {fs:.2f} Hz")

        mag_filt = butter_lowpass_filter(data=acc_data["magnitude_abs"], cutoff=20, fs=fs, order=1)
        acc_data["magnitude_filt"] = mag_filt

        peaks = find_peaks(mag_filt, height=np.mean(mag_filt) + 10*np.std(mag_filt), distance=500)
        #st.write(peaks)

        inicio = peaks[0][0]
        st.write(f"peak index: {inicio}")
        epoch_inicio = acc_data["epoch"][inicio]
        st.write(f"epoch inicio: {epoch_inicio}")
        
        target_epoch = epoch_inicio + 30
        st.write(f"target epoch: {target_epoch}")

        start_idx = (acc_data["epoch"] - target_epoch).abs().idxmin()
        start_epoch = acc_data.loc[start_idx, "epoch"]

        st.write(f"start epoch: {start_epoch}")
        st.write(f"start index: {start_idx}")

        epoch_fin = start_epoch + 60
        end_idx = (acc_data["epoch"] - epoch_fin).abs().idxmin()
        end_epoch = acc_data.loc[end_idx, "epoch"]
        st.write(f"end epoch: {end_epoch}")
        st.write(f"end index: {end_idx}")


        plot_data(acc_data["magnitude_abs"], mag_filt, acc_data["epoch"], start_epoch, end_epoch)

        acc_trim = acc_data.iloc[start_idx:end_idx]
        st.write(acc_trim)

        start_idx = (ecg_data["epoch"] - start_epoch).abs().idxmin()
        start_epoch = ecg_data.loc[start_idx, "epoch"]
        epoch_fin = start_epoch + 60
        end_idx = (ecg_data["epoch"] - epoch_fin).abs().idxmin()
        end_epoch = ecg_data.loc[end_idx, "epoch"]
        ecg_trim = ecg_data.iloc[start_idx:end_idx]
        st.write(ecg_trim)

        plot_data(ecg_data["ecg_uV"], ecg_data["ecg_uV"], ecg_data["epoch"], start_epoch, end_epoch)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.getbuffer())
            temp_path = tmp.name

        video = VideoFileClip(temp_path)
        video.audio.write_audiofile("audio.wav")

        fs_audio, audio_data = wavfile.read("audio.wav")
        st.write(fs_audio)
        time_axis = np.arange(len(audio_data)) / fs_audio

        df_audio = pd.DataFrame({
            "time": time_axis,
            "left": audio_data[:, 0],
            "right": audio_data[:, 1],
            "amplitude": np.sqrt(audio_data[:, 0]**2 + audio_data[:, 1]**2)
        })
        #st.line_chart(df_audio["amplitude"])

        #plot_data(df_audio["amplitude"], df_audio["amplitude"], df_audio["time"], 0, -1)

        target_fs = fs 
        n_samples = int(len(audio_data) * target_fs / fs_audio)
        audio_resampled = resample(audio_data, n_samples)

        time = np.arange(len(audio_resampled)) / target_fs


        df_audio_resampled = pd.DataFrame({
            "time": time,
            "left": audio_resampled[:, 0],
            "right": audio_resampled[:, 1],
            "amplitude": (audio_resampled[:, 0]+ audio_resampled[:, 1]) / 2
        })

        st.write(df_audio_resampled)
        st.line_chart(df_audio_resampled["amplitude"])

        st.audio("audio.wav")




            
    else:
        st.info("Subir archivos para realizar análisis")


def main():
    sync_data()

if __name__ == "__main__":
    main()
