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

import plotly.graph_objects as go
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
from moviepy import VideoFileClip
import tempfile
from scipy.io import wavfile
from scipy.signal import resample
import subprocess
from pathlib import Path
import zipfile
import io


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


def bandpass(data, sr, low=300, high=3000):
    nyq = 0.5 * sr
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def zip_outputs():
    outputs_dir = Path("outputs")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in outputs_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(outputs_dir))

    zip_buffer.seek(0)
    return zip_buffer


def plot_data(mag_abs, mag_filt, epoch, start_epoch, end_epoch):

    plot_df = pd.DataFrame({
        "timestamp_s": epoch,
        "abs_mag": mag_abs,
        "filtered": mag_filt
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp_s"],
            y=plot_df["abs_mag"],
            mode="lines",
            name="abs_mag"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp_s"],
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
        xaxis_title="timestamp_s",
        yaxis_title="Magnitude",
        legend_title="Signals"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_audio(energy_time, energy, cough_times, peaks):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=energy_time,
        y=energy,
        name="Energy"
    ))

    fig.add_trace(go.Scatter(
        x=cough_times,
        y=energy[peaks],
        mode="markers",
        name="Detected coughs",
        marker=dict(size=10)
    ))

    fig.update_layout(
        title="Cough detection",
        xaxis_title="Time (s)",
        yaxis_title="Energy"
    )

    st.plotly_chart(fig, use_container_width=True)

def fix_data(df):
    df["timestamp_s"] = df["timestamp_s"].str.replace(",", ".").astype(float)
    df["time"] = pd.to_datetime(df["timestamp_s"], unit="s")
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
    dt = np.diff(df["timestamp_s"])
    mean_dt = np.mean(dt)
    fs = 1/mean_dt
    return fs


def reset_dir(path):
    import shutil
    from pathlib import Path

    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def process_video(video_file, output_dir):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.getbuffer())
        temp_path = tmp.name

    video = VideoFileClip(temp_path)
    video.audio.write_audiofile(f"{output_dir}/audio.wav")

    fs_audio, audio_data = wavfile.read(f"{output_dir}/audio.wav")
    time_axis = np.arange(len(audio_data)) / fs_audio

    # df_audio = pd.DataFrame({
    #     "time": time_axis,
    #     "left": audio_data[:, 0],
    #     "right": audio_data[:, 1],
    #     "average": (audio_data[:, 0] + audio_data[:, 1])/2
    # })

    st.audio(f"{output_dir}/audio.wav")

    audio_f = bandpass((audio_data[:, 0] + audio_data[:, 1])/2, fs_audio)

    # -----------------------------
    # Short-time energy calculation
    # -----------------------------
    frame_length = int(0.02 * fs_audio)   # 20 ms
    hop = int(0.01 * fs_audio)            # 10 ms

    energy = np.array([
        np.sum(audio_f[i:i+frame_length]**2)
        for i in range(0, len(audio_f) - frame_length, hop)
    ])

    energy_time = np.arange(len(energy)) * hop / fs_audio

    # -----------------------------
    # Peak detection (coughs)
    # -----------------------------
    threshold = np.mean(energy) + 10*np.std(energy)

    peaks, properties = find_peaks(
        energy,
        height=threshold,
        distance=int(1 / (hop / fs_audio))  # minimum 1000 ms between coughs
    )

    cough_times = peaks * hop / fs_audio
    plot_audio(energy_time, energy, cough_times, peaks)

    start = cough_times[0]
    start += 30
    duration = 60

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.getbuffer())
        input_path = tmp.name

    output_path = f"{output_dir}/video.mp4"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ], check=True)

    #st.video(output_path)




def process_data(acc_file, ecg_file, output_dir):
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
    epoch_inicio = acc_data["timestamp_s"][inicio]
    st.write(f"epoch inicio: {epoch_inicio}")
    
    target_epoch = epoch_inicio + 30
    st.write(f"target epoch: {target_epoch}")

    start_idx = (acc_data["timestamp_s"] - target_epoch).abs().idxmin()
    start_epoch = acc_data.loc[start_idx, "timestamp_s"]

    st.write(f"start epoch: {start_epoch}")
    st.write(f"start index: {start_idx}")

    epoch_fin = start_epoch + 60
    end_idx = (acc_data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = acc_data.loc[end_idx, "timestamp_s"]
    st.write(f"end epoch: {end_epoch}")
    st.write(f"end index: {end_idx}")

    plot_data(acc_data["magnitude_abs"], mag_filt, acc_data["timestamp_s"], start_epoch, end_epoch)

    acc_trim = acc_data.iloc[start_idx:end_idx]
    start_idx = (ecg_data["timestamp_s"] - start_epoch).abs().idxmin()
    start_epoch = ecg_data.loc[start_idx, "timestamp_s"]
    epoch_fin = start_epoch + 60
    end_idx = (ecg_data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = ecg_data.loc[end_idx, "timestamp_s"]
    ecg_trim = ecg_data.iloc[start_idx:end_idx]

    plot_data(ecg_data["ecg_uV"], ecg_data["ecg_uV"], ecg_data["timestamp_s"], start_epoch, end_epoch)

    #st.write(acc_trim)
    #st.write(ecg_trim)

    acc_trim.to_csv(f"{output_dir}/acc_data.csv", index=False)
    ecg_trim.to_csv(f"{output_dir}/ecg_data.csv", index=False)

def sync_data():
    # Streamlit app setup
    st.title("Polar Sync 🐻‍❄️")
    st.sidebar.markdown("# Polar Sync 🐻‍❄️")

    st.write("Cargar datos obtenidos ")
    # Cargar archivos CSV
    acc_file = st.file_uploader("Cargar archivo CSV con datos de aceleración", type=["csv"])
    ecg_file = st.file_uploader("Cargar archivo CSV con datos de electrocardiograma", type=["csv"])
    video_file = st.file_uploader("Cargar archivo de video", type=["mp4"])

    if acc_file and ecg_file and video_file:
        output_dir = "outputs"
        reset_dir(output_dir)
        process_data(acc_file, ecg_file, output_dir)
        process_video(video_file, output_dir) 

        zip_data = zip_outputs()

        st.sidebar.download_button(
            label="Descargar Resultados",
            data=zip_data,
            file_name="outputs.zip",
            mime="application/zip"
        )

    else:
        st.info("Subir archivos para realizar análisis")


def main():
    sync_data()

if __name__ == "__main__":
    main()
