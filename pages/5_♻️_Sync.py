# -*- coding: utf-8 -*-
"""
@file    : 10_sync.py
@brief   : Handles synchronization of data.
@date    : 2026/03/24
@version : 1.1.0
@author  : Lucas Cortés.
"""

import io
import zipfile
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from moviepy import VideoFileClip
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------

ICON = Image.open("assets/logos/favicon.png")

st.set_page_config(
    page_title="CSV Handler",
    page_icon=ICON,
    layout="wide",
)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

OUTPUT_DIR = Path("outputs")

COUGH_PEAK_FACTOR = 10
ACC_PEAK_FACTOR = 10

TRIM_OFFSET = 30
TRIM_DURATION = 60

FRAME_LENGTH_MS = 20
HOP_MS = 10


# ---------------------------------------------------
# File Utilities
# ---------------------------------------------------

def reset_dir(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def zip_outputs():
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in OUTPUT_DIR.rglob("*"):
            if file.is_file() and file.name != "audio.wav":
                zipf.write(file, file.relative_to(OUTPUT_DIR))

    buffer.seek(0)
    return buffer


# ---------------------------------------------------
# Signal Processing
# ---------------------------------------------------

def butter_lowpass_filter(data, cutoff, fs, order=1):

    if order == 0:
        return data

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, data)


def bandpass(data, sr, low=300, high=3000):

    nyq = 0.5 * sr
    b, a = butter(4, [low / nyq, high / nyq], btype="band")

    return filtfilt(b, a, data)


def get_sampling_frequency(timestamps):

    dt = np.diff(timestamps)
    return 1 / np.mean(dt)


# ---------------------------------------------------
# Plotting
# ---------------------------------------------------

def plot_data(signal, epoch, start_epoch, end_epoch, peaks=None, title="Signal"):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=signal,
            mode="lines",
            name="Signal",
        )
    )

    if peaks is not None:

        fig.add_trace(
            go.Scatter(
                x=epoch[peaks],
                y=signal[peaks],
                mode="markers",
                marker=dict(size=10),
                name="Peaks",
            )
        )

    fig.add_vline(x=start_epoch, line_dash="dash", line_color="green")
    fig.add_vline(x=end_epoch, line_dash="dash", line_color="red")

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# Data Preparation
# ---------------------------------------------------

def fix_data(df):

    df = df.dropna()

    if "timestamp_s" in df.columns:

        df["timestamp"] = df["timestamp_s"].str.replace(",", ".").astype(float)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    elif "TimeStamp" in df.columns:

        df["timestamp"] = df["TimeStamp"].astype(float)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    else:
        st.error("No timestamp column found.")
        st.stop()

    # Move timestamp and time to the beginning
    cols = ["timestamp", "datetime"] + [c for c in df.columns if c not in ["timestamp", "datetime"]]
    df = df[cols]

    df = df.drop(columns=["timestamp_s", "TimeStamp", "DateTime"], errors="ignore")

    return df


def compute_magnitude(df):

    x = df["x_mG"].values
    y = df["y_mG"].values
    z = df["z_mG"].values

    mag = np.sqrt(x**2 + y**2 + z**2)

    df["magnitude"] = mag
    df["magnitude_abs"] = np.abs(mag)
    df["magnitude_dc"] = np.abs(mag - np.mean(mag))

    fs = get_sampling_frequency(df["timestamp"])

    df["magnitude_filt"] = butter_lowpass_filter(
        df["magnitude_dc"],
        cutoff=20,
        fs=fs,
    )

    return df


# ---------------------------------------------------
# ACC Processing
# ---------------------------------------------------

def process_data_acc(file):

    data = pd.read_csv(file, sep=";")
    data = fix_data(data)
    data = compute_magnitude(data)

    signal = data["magnitude_filt"].values

    threshold = np.mean(signal) + ACC_PEAK_FACTOR * np.std(signal)

    peaks, _ = find_peaks(signal, height=threshold, distance=500)

    start_epoch = data["timestamp"].iloc[peaks[0]] + TRIM_OFFSET
    end_epoch = start_epoch + TRIM_DURATION

    start_idx = (data["timestamp"] - start_epoch).abs().idxmin()
    end_idx = (data["timestamp"] - end_epoch).abs().idxmin()

    trimmed = data.iloc[start_idx:end_idx]

    trimmed.to_csv(OUTPUT_DIR / "acc_data.csv", index=False)

    plot_data(
        signal,
        data["datetime"].values,
        pd.to_datetime(start_epoch, unit="s"),
        pd.to_datetime(end_epoch, unit="s"),
        peaks,
        "ACC Peak Detection",
    )

    return start_epoch


# ---------------------------------------------------
# ECG Processing
# ---------------------------------------------------

def process_data_ecg(file, start_epoch):

    data = pd.read_csv(file, sep=";")
    data = fix_data(data)

    end_epoch = start_epoch + TRIM_DURATION

    start_idx = (data["timestamp"] - start_epoch).abs().idxmin()
    end_idx = (data["timestamp"] - end_epoch).abs().idxmin()

    trimmed = data.iloc[start_idx:end_idx]

    trimmed.to_csv(OUTPUT_DIR / "ecg_data.csv", index=False)

    plot_data(
        data["ecg_uV"],
        data["datetime"].values,
        pd.to_datetime(start_epoch, unit="s"),
        pd.to_datetime(end_epoch, unit="s"),
        title="ECG Signal",
    )


# ---------------------------------------------------
# CONTEC Processing
# ---------------------------------------------------

def process_data_contec(file, start_epoch):

    data = pd.read_csv(file)
    data = fix_data(data)

    end_epoch = start_epoch + TRIM_DURATION

    start_idx = (data["timestamp"] - start_epoch).abs().idxmin()
    end_idx = (data["timestamp"] - end_epoch).abs().idxmin()

    trimmed = data.iloc[start_idx:end_idx]

    trimmed.to_csv(OUTPUT_DIR / "contec_data.csv", index=False)

    plot_data(
        data["HR"],
        data["datetime"].values,
        pd.to_datetime(start_epoch, unit="s"),
        pd.to_datetime(end_epoch, unit="s"),
        title="CONTEC HR",
    )


# ---------------------------------------------------
# Video Processing
# ---------------------------------------------------

def process_video(video_file, start_epoch):

    # --------------------------------------------------
    # Save uploaded video to temp file
    # --------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.getbuffer())
        video_path = tmp.name

    video = VideoFileClip(video_path)
    audio_path = OUTPUT_DIR / "audio.wav"

    # --------------------------------------------------
    # Extract audio
    # --------------------------------------------------
    video.audio.write_audiofile(audio_path)

    fs_audio, audio_data = wavfile.read(audio_path)

    # Convert stereo → mono
    audio_avg = audio_data.mean(axis=1)

    # Bandpass filter for cough frequencies
    audio_f = bandpass(audio_avg, fs_audio)

    # --------------------------------------------------
    # Short-time energy
    # --------------------------------------------------
    frame_len = int(FRAME_LENGTH_MS / 1000 * fs_audio)
    hop = int(HOP_MS / 1000 * fs_audio)

    energy = np.array([
        np.sum(audio_f[i:i + frame_len] ** 2)
        for i in range(0, len(audio_f) - frame_len, hop)
    ])

    energy_time = np.arange(len(energy)) * hop / fs_audio

    # --------------------------------------------------
    # Peak detection (cough)
    # --------------------------------------------------
    threshold = np.mean(energy) + COUGH_PEAK_FACTOR * np.std(energy)

    peaks, _ = find_peaks(
        energy,
        height=threshold,
        distance=int(fs_audio / hop),
    )

    cough_times = peaks * hop / fs_audio

    if len(cough_times) == 0:
        st.error("No cough detected in audio.")
        return

    # --------------------------------------------------
    # Synchronization
    # --------------------------------------------------
    # ACC cough = start_epoch - TRIM_OFFSET
    # Align audio cough to that moment
    video_start_epoch = (start_epoch - TRIM_OFFSET) - cough_times[0]

    # Build datetime vector for plotting
    energy_timestamp = video_start_epoch + energy_time
    energy_datetime = pd.to_datetime(energy_timestamp, unit="s")

    # --------------------------------------------------
    # Compute trim window
    # --------------------------------------------------
    start = cough_times[0] + TRIM_OFFSET
    end = start + TRIM_DURATION

    start_dt = pd.to_datetime(video_start_epoch + start, unit="s")
    end_dt = pd.to_datetime(video_start_epoch + end, unit="s")

    # --------------------------------------------------
    # Trim video
    # --------------------------------------------------
    output_video = OUTPUT_DIR / "video.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(TRIM_DURATION),
            "-c", "copy",
            output_video,
        ],
        check=True,
    )

    # --------------------------------------------------
    # Plot cough detection
    # --------------------------------------------------
    plot_data(
        energy,
        energy_datetime,
        start_dt,
        end_dt,
        peaks,
        "Cough Detection",
    )


# ---------------------------------------------------
# UI
# ---------------------------------------------------

def sync_data():

    st.title("Polar Sync 🐻‍❄️")
    st.sidebar.markdown("# Polar Sync 🐻‍❄️")

    reset_dir(OUTPUT_DIR)

    col1, col2, col3, col4 = st.columns(4)

    acc_valid = ecg_valid = contec_valid = video_valid = False

    # ACC
    with col1:

        acc_file = st.file_uploader("ACC CSV", type=["csv"], key="acc")

        if acc_file:

            if "acc" not in acc_file.name.lower():
                st.error("ACC ❌")
            else:
                st.success("ACC ✅")
                start_epoch = process_data_acc(acc_file)
                acc_valid = True

    # ECG
    with col2:

        ecg_file = st.file_uploader(
            "ECG CSV",
            type=["csv"],
            key="ecg",
            disabled=not acc_valid,
        )

        if ecg_file:

            if "ecg" not in ecg_file.name.lower():
                st.error("ECG ❌")
            else:
                st.success("ECG ✅")
                process_data_ecg(ecg_file, start_epoch)
                ecg_valid = True

    # CONTEC
    with col3:

        contec_file = st.file_uploader(
            "CONTEC CSV",
            type=["csv"],
            key="contec",
            disabled=not (acc_valid and ecg_valid),
        )

        if contec_file:

            if "contec" not in contec_file.name.lower():
                st.error("CONTEC ❌")
            else:
                st.success("CONTEC ✅")
                process_data_contec(contec_file, start_epoch)
                contec_valid = True

    # VIDEO
    with col4:

        video_file = st.file_uploader(
            "Video MP4",
            type=["mp4"],
            key="video",
            disabled=not (acc_valid and ecg_valid and contec_valid),
        )

        if video_file:

            st.success("VIDEO ✅")
            process_video(video_file, start_epoch)
            video_valid = True

    if acc_valid and ecg_valid and contec_valid and video_valid:

        zip_data = zip_outputs()

        st.sidebar.download_button(
            "Descargar Resultados",
            data=zip_data,
            file_name="outputs.zip",
            mime="application/zip",
        )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    sync_data()


if __name__ == "__main__":
    main()