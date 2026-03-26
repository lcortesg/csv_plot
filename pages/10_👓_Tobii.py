# -*- coding: utf-8 -*-
"""
@file    : 10_👓_tobii.py
@brief   : Handles Tobii processing.
@date    : 2025/09/09
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from dtaidistance import dtw, dtw_visualisation as dtwvis
from scipy.signal import find_peaks

im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)


def merge_eyes(left_df, right_df, on="frame", suffixes=("_L", "_R")):
    """
    Merge left and right eye DataFrames on a common column,
    adding suffixes to overlapping column names.

    Parameters
    ----------
    left_df : pd.DatFrame
        DataFrame for the left eye.
    right_df : pd.DataFrame
        DataFrame for the right eye.
    on : str
        Column name to merge on (default: "frame").
    suffixes : tuple
        Suffixes for left and right DataFrames (default: ("_L", "_R")).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with suffixed column names.
    """
    return pd.merge(left_df, right_df, on=on, suffixes=suffixes)


def data_cleaning(data):
    sensors = data["Sensor"].dropna().unique()
    filter = st.selectbox(
        "Select sensor",
        sensors,
    )
    data = data[data["Sensor"] == filter]
    return data


def subsample_to_match(short_df, long_df):
    """
    Subsample the longer DataFrame to match the number of rows of the shorter one.

    Parameters
    ----------
    short_df : pd.DataFrame
        Reference DataFrame with the desired number of rows.
    long_df : pd.DataFrame
        DataFrame to be subsampled.

    Returns
    -------
    pd.DataFrame
        Subsampled DataFrame with the same number of rows as short_df.
    """
    n_rows = len(short_df)
    long_len = len(long_df)

    if long_len <= n_rows:
        # Already short enough
        return long_df.copy()

    # Choose evenly spaced indices
    indices = np.linspace(0, long_len - 1, n_rows).astype(int)
    return long_df.iloc[indices].reset_index(drop=True)


def data_extraction(dataJ, dataN):
    """
    Extract left/right eye signals from juvenile and target data,
    flatten to 1-D arrays, and interpolate missing values.
    """

    def interpolate(arr):
        return pd.Series(arr).interpolate(method="linear", limit_direction="both").to_numpy()

    def demean(arr):
        return arr - np.mean(arr)

    def normalize(arr):
        # Normalize to unit variance
        std = np.std(arr)
        if std != 0:
            arr_normalized = arr / std
        else:
            arr_normalized = arr  # if std=0, leave as is

        return arr_normalized

    def flatten(series, int=True, filt=True, norm=True):
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        arr = np.asarray(series).flatten()

        if int:
            arr = interpolate(arr)

        if filt:
            arr = demean(arr)

        if norm:
            arr = normalize(arr)

        return arr

    # Left eye
    jplx = flatten(dataJ["X_FILT_L"])
    jply = flatten(dataJ["Y_FILT_L"])
    # Right eye
    jprx = flatten(dataJ["X_FILT_R"])
    jpry = flatten(dataJ["Y_FILT_R"])

    # Target signals
    tplx = flatten(dataN["Pupil position left X"])
    tply = flatten(dataN["Pupil position left Y"])
    tprx = flatten(dataN["Pupil position right X"])
    tpry = flatten(dataN["Pupil position right Y"])

    return jplx, jply, jprx, jpry, tplx, tply, tprx, tpry


def data_analysis(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry):
    """Perform DTW analysis and visualization for left/right eye signals."""
    dlx = dtw.distance(jplx, tplx)
    plx = dtw.warping_path(jplx, tplx)
    figure, axes = dtwvis.plot_warping(jplx, tplx, plx)
    mean_dist = dlx / len(jplx)
    st.subheader(f"Left eye, X coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    dly = dtw.distance(jply, tply)
    ply = dtw.warping_path(jply, tply)
    figure, axes = dtwvis.plot_warping(jply, tply, ply)
    mean_dist = dly / len(jply)
    st.subheader(f"Left eye, Y coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    drx = dtw.distance(jprx, tprx)
    prx = dtw.warping_path(jprx, tprx)
    figure, axes = dtwvis.plot_warping(jprx, tprx, prx)
    mean_dist = drx / len(jprx)
    st.subheader(f"Right eye, X coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    dry = dtw.distance(jpry, tpry)
    pry = dtw.warping_path(jpry, tpry)
    figure, axes = dtwvis.plot_warping(jpry, tpry, pry)
    mean_dist = dry / len(jpry)
    st.subheader(f"Right eye, Y coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    # Compute DTW distances and warping paths for left/right eye signals
    # using FastDTW.

    # dlx, plx = fastdtw(jplx, tplx, dist=euclidean)
    # dly, ply = fastdtw(jply, tply, dist=euclidean)

    # drx, prx = fastdtw(jprx, tprx, dist=euclidean)
    # dry, pry = fastdtw(jpry, tpry, dist=euclidean)


def histograma(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry):
    """
    Genera histogramas comparativos entre las señales de los ojos y las señales objetivo.
    Además, muestra la PDF gaussiana ajustada para cada señal.
    """
    señales = [
        ("Ojo Izquierdo X", jplx, tplx),
        ("Ojo Izquierdo Y", jply, tply),
        ("Ojo Derecho X", jprx, tprx),
        ("Ojo Derecho Y", jpry, tpry),
    ]

    for nombre, señal1, señal2 in señales:
        fig, ax = plt.subplots()
        # Histograma
        ax.hist(señal1, bins=30, alpha=0.5, label="Algoritmo", color="blue", density=True)
        ax.hist(señal2, bins=30, alpha=0.5, label="Tobii pro", color="orange", density=True)

        # PDF gaussiana para señal1
        mu1, std1 = np.mean(señal1), np.std(señal1)
        x1 = np.linspace(np.min(señal1), np.max(señal1), 100)
        ax.plot(
            x1,
            norm.pdf(x1, mu1, std1),
            color="blue",
            linestyle="--",
            label="PDF Algoritmo",
        )

        # PDF gaussiana para señal2
        mu2, std2 = np.mean(señal2), np.std(señal2)
        x2 = np.linspace(np.min(señal2), np.max(señal2), 100)
        ax.plot(
            x2,
            norm.pdf(x2, mu2, std2),
            color="orange",
            linestyle="--",
            label="PDF Tobii pro",
        )

        ax.set_title(f"Histograma y PDF - {nombre}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)


def seleccionar_primer_peak(f, spec, min_freq=0.2, umbral_rel=0.1):
    """
    Selecciona el primer peak significativo (más cercano al DC pero no en DC).

    Parámetros:
    ------------
    f : array
        Vector de frecuencias.
    spec : array
        Espectro (magnitud o PSD).
    min_freq : float
        Frecuencia mínima para ignorar el DC y ruido de muy baja frecuencia (Hz).
    umbral_rel : float
        Umbral relativo respecto al valor máximo del espectro (0–1).
        Ej: 0.1 -> considera solo picos >10% del máximo.
    """
    # Filtrar la parte positiva del espectro
    mask = f >= min_freq
    f_pos = f[mask]
    spec_pos = spec[mask]

    # Buscar picos sobre el umbral relativo
    height_min = np.max(spec_pos) * umbral_rel
    peaks, props = find_peaks(spec_pos, height=height_min)

    if len(peaks) == 0:
        return np.nan  # no se encontró nada

    # Elegir el PRIMER pico (menor frecuencia)
    primer_pico_idx = peaks[0]
    freq_primer_pico = f_pos[primer_pico_idx]
    return freq_primer_pico


def fft_comparacion(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry, fs_tobi=100, fs_al=30):
    """
    Calcula y grafica la FFT de las señales comparadas para analizar sus componentes en frecuencia.
    Además, muestra la frecuencia fundamental de cada señal.
    fs_tobi: frecuencia de muestreo (Hz) tobii pro.
    fs_al: frecuencia de muestreo (Hz) algoritmo.
    """
    señales = [
        ("Ojo Izquierdo X", jplx, tplx),
        ("Ojo Izquierdo Y", jply, tply),
        ("Ojo Derecho X", jprx, tprx),
        ("Ojo Derecho Y", jpry, tpry),
    ]

    umbral_relativo = 0.1  # 10% del pico máximo, ajustable
    min_freq = 0.2  # ignorar todo por debajo de 0.2 Hz

    # señal 1: algoritmo, señal 2: tobii
    for nombre, señal1, señal2 in señales:
        N = min(len(señal1), len(señal2))

        # FFT de las señales
        f_1 = fftfreq(N, 1 / fs_al)[: N // 2]
        f_2 = fftfreq(N, 1 / fs_tobi)[: N // 2]
        fft1 = np.abs(fft(señal1[:N]))[: N // 2]
        fft2 = np.abs(fft(señal2[:N]))[: N // 2]

        # Ignorar la componente DC (0 Hz) y buscar el peak mas significativo siguiente.
        freq_fund1 = seleccionar_primer_peak(f_1, fft1, min_freq=min_freq, umbral_rel=umbral_relativo)
        freq_fund2 = seleccionar_primer_peak(f_2, fft2, min_freq=min_freq, umbral_rel=umbral_relativo)

        # Parte original
        # idx1 = np.argmax(fft1[1:]) + 1
        # idx2 = np.argmax(fft2[1:]) + 1
        # freq_fund1 = f_1[idx1]
        # freq_fund2 = f_2[idx2]

        # Graficos FFT
        fig, ax = plt.subplots()
        ax.plot(f_1, fft1, label="Algoritmo", color="blue")
        ax.plot(f_2, fft2, label="Tobii pro", color="orange")

        if not np.isnan(freq_fund1):
            ax.axvline(freq_fund1, color="blue", linestyle="--", alpha=0.6)
        if not np.isnan(freq_fund2):
            ax.axvline(freq_fund2, color="orange", linestyle="--", alpha=0.6)

        ax.set_title(f"FFT comparativa - {nombre}")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.legend()
        st.pyplot(fig)

        # st.info(
        #    f"Frecuencia fundamental {nombre}: "
        #    f"Algoritmo = {freq_fund1:.2f} Hz, "
        #    f"Tobii pro = {freq_fund2:.2f} Hz"
        # )

        # ---- Mostrar frecuencias ----
        if np.isnan(freq_fund1) or np.isnan(freq_fund2):
            st.warning(f"No se pudo estimar la frecuencia fundamental para {nombre}.")
        else:
            st.info(
                f"Frecuencia fundamental {nombre}: "
                f"Algoritmo = {freq_fund1:.2f} Hz, "
                f"Tobii pro = {freq_fund2:.2f} Hz"
            )


def tobii_comp():
    # Streamlit app setup
    st.title("Análisis Tobii 👓")
    st.sidebar.markdown("# Análisis Tobii 👓")

    st.write("1. Cargar datos procesados del algoritmo:")
    # Cargar archivos CSV
    datos_josefa_L = st.file_uploader("Cargar archivo CSV con datos pupila izquierda", type=["csv"])
    datos_josefa_R = st.file_uploader("Cargar archivo CSV con datos pupilas derecha", type=["csv"])

    if datos_josefa_L and datos_josefa_R:
        data_L = pd.read_csv(datos_josefa_L)
        data_R = pd.read_csv(datos_josefa_R)
        data_J = merge_eyes(data_L, data_R, on="Frame", suffixes=("_L", "_R"))
        # Upload CSV file
        st.write("2. Cargar datos procesados con Tobii:")
        uploaded_file = st.file_uploader("Cargar archivo XLSM con datos Tobii", type=["xlsm"])

        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1]
            if ext == "csv":
                data = pd.read_csv(uploaded_file)
            elif ext == "xlsm":
                data = pd.read_excel(uploaded_file, engine="openpyxl")
            data = data_cleaning(data)  # Filtrar datos por sensor
            data_N = subsample_to_match(data_J, data)  # Subsamplear para igualar tamaño

            # Display the raw data
            if st.sidebar.toggle("Mostrar datos"):
                st.write("### Raw Data")
                st.write(data_J)
                st.write(data_N)

            st.write("3. Análisis de datos")
            with st.spinner("Analizando los datos, por favor espere..."):
                # Extraer señales de ojos
                jplx, jply, jprx, jpry, tplx, tply, tprx, tpry = data_extraction(
                    data_J, data_N
                )  # Extraer señales de ojos
                st.write("Comparación con DTW:")
                data_analysis(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry)  # Análisis de datos
                # Comparación con histograma y FFT
                st.write("Comparación con Histograma:")
                histograma(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry)
                st.write("Comparación con FFT:")
                fft_comparacion(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry)

    else:
        st.info("Subir archivos para realizar análisis")


def main():
    tobii_comp()


if __name__ == "__main__":
    main()
