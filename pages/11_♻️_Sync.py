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
            if file.is_file() and file.name != "audio.wav":
                zipf.write(file, file.relative_to(outputs_dir))

    zip_buffer.seek(0)
    return zip_buffer


def plot_data(mag, epoch, start_epoch, end_epoch, peaks=None):

    plot_df = pd.DataFrame({
        "timestamp_s": epoch,
        "mag": mag,
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp_s"],
            y=plot_df["mag"],
            mode="lines",
            name="Magnitude"
        )
    )
    title = "Trimmed signal" if peaks is None else "Peak detection"
    if peaks is not None:
        fig.add_trace(go.Scatter(
            x=epoch[peaks],
            y=mag[peaks],
            mode="markers",
            name="Peaks",
            marker=dict(size=10)
        ))

    # Vertical lines
    fig.add_vline(x=start_epoch, line_dash="dash", line_color="green")
    fig.add_vline(x=end_epoch, line_dash="dash", line_color="red")

    fig.update_layout(
        title=title,
        xaxis_title="timestamp_s",
        yaxis_title="Magnitude",
        legend_title="Signals"
    )

    #st.plotly_chart(fig, use_container_width=True)
    return fig


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
    audio_avg = (audio_data[:, 0] + audio_data[:, 1])/2
    # time_axis = np.arange(len(audio_data)) / fs_audio

    # df_audio = pd.DataFrame({
    #     "time": time_axis,
    #     "left": audio_data[:, 0],
    #     "right": audio_data[:, 1],
    #     "average": (audio_data[:, 0] + audio_data[:, 1])/2
    # })

    #st.audio(f"{output_dir}/audio.wav")

    audio_f = bandpass(audio_avg, fs_audio)

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
    
    start = cough_times[0]
    start += 30
    duration = 60
    end = start + duration

    vid = plot_data(energy, energy_time, start, end, peaks)

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

    st.plotly_chart(vid, use_container_width=True)


def process_data(acc_file, ecg_file, output_dir):
    acc_data = pd.read_csv(acc_file, sep=";")
    ecg_data = pd.read_csv(ecg_file, sep=";")
    ecg_data = fix_data(ecg_data)
    acc_data = fix_data(acc_data)
    acc_data = get_magnitudes(acc_data)

    fs = get_sf(acc_data)
    #st.write(f"Frecuencia de muestreo estimada: {fs:.2f} Hz")

    mag_filt = butter_lowpass_filter(data=acc_data["magnitude_abs"], cutoff=20, fs=fs, order=1)
    acc_data["magnitude_filt"] = mag_filt

    peaks, properties = find_peaks(mag_filt, height=np.mean(mag_filt) + 10*np.std(mag_filt), distance=500)

    inicio = peaks[0]
    #st.write(f"peak index: {inicio}")
    epoch_inicio = acc_data["timestamp_s"][inicio]
    #st.write(f"epoch inicio: {epoch_inicio}")
    
    target_epoch = epoch_inicio + 30
    #st.write(f"target epoch: {target_epoch}")

    start_idx = (acc_data["timestamp_s"] - target_epoch).abs().idxmin()
    start_epoch = acc_data.loc[start_idx, "timestamp_s"]

    #st.write(f"start epoch: {start_epoch}")
    #st.write(f"start index: {start_idx}")

    epoch_fin = start_epoch + 60
    end_idx = (acc_data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = acc_data.loc[end_idx, "timestamp_s"]

    acc_trim = acc_data.iloc[start_idx:end_idx]
    start_idx = (ecg_data["timestamp_s"] - start_epoch).abs().idxmin()
    start_epoch = ecg_data.loc[start_idx, "timestamp_s"]
    epoch_fin = start_epoch + 60
    end_idx = (ecg_data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = ecg_data.loc[end_idx, "timestamp_s"]
    ecg_trim = ecg_data.iloc[start_idx:end_idx]
    acc_trim.to_csv(f"{output_dir}/acc_data.csv", index=False)
    ecg_trim.to_csv(f"{output_dir}/ecg_data.csv", index=False)


    acc = plot_data(mag_filt, acc_data["timestamp_s"], start_epoch, end_epoch, peaks)
    ecg = plot_data(ecg_data["ecg_uV"], ecg_data["timestamp_s"], start_epoch, end_epoch)

    return acc, ecg


def process_data_acc(file, output_dir):
    data = pd.read_csv(file, sep=";")
    data = fix_data(data)
    data = get_magnitudes(data)

    fs = get_sf(data)

    mag_filt = butter_lowpass_filter(data=data["magnitude_abs"], cutoff=20, fs=fs, order=1)
    data["magnitude_filt"] = mag_filt

    peaks, properties = find_peaks(mag_filt, height=np.mean(mag_filt) + 10*np.std(mag_filt), distance=500)

    inicio = peaks[0]
    epoch_inicio = data["timestamp_s"][inicio]
    target_epoch = epoch_inicio + 30
    start_idx = (data["timestamp_s"] - target_epoch).abs().idxmin()
    start_epoch = data.loc[start_idx, "timestamp_s"]

    epoch_fin = start_epoch + 60
    end_idx = (data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = data.loc[end_idx, "timestamp_s"]

    acc_trim = data.iloc[start_idx:end_idx]
    acc_trim.to_csv(f"{output_dir}/acc_data.csv", index=False)
    acc = plot_data(mag_filt, data["timestamp_s"], start_epoch, end_epoch, peaks)
   
    st.plotly_chart(acc, use_container_width=True)

    return start_epoch


def process_data_ecg(file, output_dir, start_epoch):
    data = pd.read_csv(file, sep=";")
    data = fix_data(data)
    start_idx = (data["timestamp_s"] - start_epoch).abs().idxmin()
    start_epoch = data.loc[start_idx, "timestamp_s"]
    epoch_fin = start_epoch + 60
    end_idx = (data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = data.loc[end_idx, "timestamp_s"]
    ecg_trim = data.iloc[start_idx:end_idx]
    ecg_trim.to_csv(f"{output_dir}/ecg_data.csv", index=False)
    ecg = plot_data(data["ecg_uV"], data["timestamp_s"], start_epoch, end_epoch)
    st.plotly_chart(ecg, use_container_width=True)




def process_data_contec(file, output_dir, start_epoch):
    data = pd.read_csv(file, sep=";")
    data = fix_data(data)
    start_idx = (data["timestamp_s"] - start_epoch).abs().idxmin()
    start_epoch = data.loc[start_idx, "timestamp_s"]
    epoch_fin = start_epoch + 60
    end_idx = (data["timestamp_s"] - epoch_fin).abs().idxmin()
    end_epoch = data.loc[end_idx, "timestamp_s"]
    ecg_trim = data.iloc[start_idx:end_idx]
    ecg_trim.to_csv(f"{output_dir}/contec_data.csv", index=False)
    ecg = plot_data(data["ecg_uV"], data["timestamp_s"], start_epoch, end_epoch)
    st.plotly_chart(ecg, use_container_width=True)


def sync_data():
    st.title("Polar Sync 🐻‍❄️")
    st.sidebar.markdown("# Polar Sync 🐻‍❄️")
    output_dir = "outputs"
    reset_dir(output_dir)

    col1, col2, col3, col4 = st.columns(4)

    # --- ACC ---
    with col1:
        acc_file = st.file_uploader("ACC CSV", type=["csv"], key="acc")

        acc_valid = False
        if acc_file:
            if "acc" not in acc_file.name.lower():
                st.error("Debe contener 'acc'")
            else:
                st.success("ACC ✓")
                start_epoch = process_data_acc(acc_file, output_dir)
                acc_valid = True


    # --- ECG ---
    with col2:
        ecg_file = None
        ecg_valid = False

        if acc_valid:
            ecg_file = st.file_uploader("ECG CSV", type=["csv"], key="ecg")

            if ecg_file:
                if "ecg" not in ecg_file.name.lower():
                    st.error("Debe contener 'ecg'")
                else:
                    st.success("ECG ✓")
                    process_data_ecg(ecg_file, output_dir, start_epoch)
                    ecg_valid = True
        else:
            st.info("Subir ACC")
    with col3:
        contec_file = None
        contec_valid = False

        if acc_valid and ecg_valid:
            contec_file = st.file_uploader("CONTEC CSV", type=["csv"], key="contec")

            if contec_file:
                if "contec" not in contec_file.name.lower():
                    st.error("Debe contener 'contec'")
                else:
                    st.success("CONTEC ✓")
                    process_data_contec(contec_file, output_dir, start_epoch)
                    contec_valid = True

        else:
            st.info("Subir ACC y ECG")

    # --- VIDEO ---
    with col4:
        video_file = None
        video_valid = False

        if acc_valid and ecg_valid and contec_valid:
            video_file = st.file_uploader("Video MP4", type=["mp4"], key="video")

            if video_file:
                st.success("Video ✓")
                process_video(video_file, output_dir)
                video_valid = True
        else:
            st.info("Subir ACC, ECG y CONTEC")

    # --- Processing ---
    if acc_valid and ecg_valid and video_valid:

        zip_data = zip_outputs()

        st.sidebar.download_button(
            label="Descargar Resultados",
            data=zip_data,
            file_name="outputs.zip",
            mime="application/zip"
        )


def main():
    sync_data()

if __name__ == "__main__":
    main()
