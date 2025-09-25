# -*- coding: utf-8 -*-
"""
@file    : 10_👓_tobii.py
@brief   : Handles Tobii processing.
@date    : 2025/09/09
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
        return pd.Series(arr).interpolate(method='linear', limit_direction='both').to_numpy()
    
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
            series = series.iloc[:,0]
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
    """
    Perform DTW analysis and visualization for left/right eye signals.
    """
    dlx = dtw.distance(jplx, tplx)
    plx = dtw.warping_path(jplx, tplx)
    figure, axes = dtwvis.plot_warping(jplx, tplx, plx)
    mean_dist = dlx/len(jplx)
    st.subheader(f"Left eye, X coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    dly = dtw.distance(jply, tply)
    ply = dtw.warping_path(jply, tply)
    figure, axes = dtwvis.plot_warping(jply, tply, ply)
    mean_dist = dly/len(jply)
    st.subheader(f"Left eye, Y coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    drx = dtw.distance(jprx, tprx)
    prx = dtw.warping_path(jprx, tprx)
    figure, axes = dtwvis.plot_warping(jprx, tprx, prx)
    mean_dist = drx/len(jprx)
    st.subheader(f"Right eye, X coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    dry = dtw.distance(jpry, tpry)
    pry = dtw.warping_path(jpry, tpry)
    figure, axes = dtwvis.plot_warping(jpry, tpry, pry)
    mean_dist = dry/len(jpry)
    st.subheader(f"Right eye, Y coordinate, mean DTW: {mean_dist}", divider=True)
    st.pyplot(figure)

    """
    Compute DTW distances and warping paths for left/right eye signals
    using FastDTW.

    dlx, plx = fastdtw(jplx, tplx, dist=euclidean)
    dly, ply = fastdtw(jply, tply, dist=euclidean)

    drx, prx = fastdtw(jprx, tprx, dist=euclidean)
    dry, pry = fastdtw(jpry, tpry, dist=euclidean)"""

    # Agregar análisis adicionales: Histograma por cada ojo y coordenada. FFT de cada ojo y coordenada.




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
            data = data_cleaning(data) # Filtrar datos por sensor
            data_N = subsample_to_match(data_J, data) # Subsamplear para igualar tamaño
            
            # Display the raw data
            if st.sidebar.toggle("Mostrar datos"):
                st.write("### Raw Data")
                st.write(data_J)
                st.write(data_N)

            st.write("3. Analizando los datos...")
            with st.spinner("Analizando los datos, por favor espere..."):
                # Extraer señales de ojos
                jplx, jply, jprx, jpry, tplx, tply, tprx, tpry = data_extraction(data_J, data_N) # Extraer señales de ojos
                data_analysis(jplx, jply, jprx, jpry, tplx, tply, tprx, tpry) # Análisis de datos
            
    else:
        st.info("Subir archivos para realizar análisis")


def main():
    tobii_comp()

if __name__ == "__main__":
    main()
