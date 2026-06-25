# -*- coding: utf-8 -*-
"""
@file    : 3_🫀_HRV.py
@brief   : Handles HRV processing.
@date    : 2024/08/22
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import math
import warnings

import numpy as np
import polars as pl
import heartpy as hp
import streamlit as st
import pyhrv.time_domain as td
import plotly.graph_objects as go
import pyhrv.frequency_domain as fd
import neurokit2 as nk

from PIL import Image
from hrvanalysis import get_time_domain_features, get_frequency_domain_features

im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)


def data_cleaning(data):
    data.replace(0, np.nan, inplace=True)
    if st.sidebar.toggle("Mostrar información"):
        st.write("### Información")
        f2r = data.head(1)
        for i in f2r.keys():
            # Check if the value is a number and not NaN
            value = f2r[i].values[0]
            # Skip iteration if the value is NaN
            if (isinstance(value, (int, float)) and not math.isnan(value)) or isinstance(value, (str)):
                st.markdown(f"""
                    **{i}**: {value}\n
                """)

    if "HR" not in data.columns:
        new_column_names = data.iloc[1]
        # Skip the first row and set the new column names
        data = data[2:]  # Skip the first row
        data.columns = new_column_names  # Set new column names
        # data = pd.read_csv(uploaded_file, skiprows=2)
        data.rename(columns={"HR (BPM)": "HR"}, inplace=True)
    return data


def identify_columns(data):
    hr_column = None
    time_column = None
    for col in data.columns:
        if "HR" in col or "BPM" in col:
            hr_column = col
        if "Time" in col or "Timestamp" in col:
            time_column = col
        if "Temperatures" in col:
            temp_column = col
    return hr_column, time_column, temp_column


def data_extraction(data):
    hr_column, time_column, temp_column = identify_columns(data)
    if hr_column in data.columns:
        hrvalues = data[hr_column]  # .values
        hrvalues = [int(x) for x in hrvalues if not math.isnan(x)]
        rr_intervals = [60000 / x for x in hrvalues]
    ts = list(range(len(hrvalues)))
    if time_column in data.columns:
        ts = data[time_column]  # .values
    showTemp = False
    temp = []
    if temp_column in data.columns:
        temp = data[temp_column]  # .values
        showTemp = st.sidebar.toggle("¿Mostrar temperaturas?")
    return hrvalues, rr_intervals, ts, temp, showTemp



def data_plot(hrvalues, rr_intervals, ts, temp, showTemp):
    st.write("### BPM & Temp")
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=hrvalues, mode="lines+markers", name="BPM"))
    fig.update_layout(
        title="Pulse Over Time",
        xaxis_title="Time (s)",  # Replace with appropriate unit for 'ts'
        yaxis_title="Pulse (BPM)",
        template="plotly_white",  # Optional: Set a clean background style
    )
    if showTemp:
        fig.add_trace(go.Scatter(x=ts, y=temp, mode="lines+markers", name="Temp"))
    st.plotly_chart(fig)

    st.write("### Intervalos RR")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=rr_intervals, mode="lines+markers", name="RR"))
    fig.update_layout(
        title="RR Intervals Over Time",
        xaxis_title="Time (s)",  # Replace with appropriate unit for 'ts'
        yaxis_title="RR Interval (ms)",
        template="plotly_white",  # Optional: Set a clean background style
    )
    st.plotly_chart(fig)



def get_results(method, rr_intervals):

    try:
        with warnings.catch_warnings(record=True) as W:
            # Time-domain HRV metrics
            if method == "PYHRV":
                url = "https://pyhrv.readthedocs.io/en/latest/"
                time_domain_results = td.time_domain(rr_intervals)
                frequency_domain_results = fd.welch_psd(rr_intervals, show=False)
                return time_domain_results, frequency_domain_results
            if method == "HRV-ANALYSIS":
                url = "https://aura-healthcare.github.io/hrv-analysis/"
                time_domain_results = get_time_domain_features(rr_intervals)
                frequency_domain_results = get_frequency_domain_features(rr_intervals)
                return time_domain_results, frequency_domain_results
            if method == "HEARTPY":
                url = "https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/"
                working_data, measures = hp.process_rr(rr_intervals)
                time_domain_results = measures
                frequency_domain_results = get_frequency_domain_features(
                    rr_intervals
                )  # hp.hrv(working_data, sample_rate=1.0)
                return time_domain_results, frequency_domain_results
            if method == "NEUROKIT2":
                url = "https://neuropsychology.github.io/NeuroKit/"
                peaks = nk.intervals_to_peaks(rr_intervals)
                time_domain_results = nk.hrv_time(peaks)
                frequency_domain_results = nk.hrv_frequency(peaks)
                return time_domain_results, frequency_domain_results
            st.markdown(
                f"Consultar más información sobre la librería utilizada en el siguiente link: [{method}](%s)"
                % url
            )
            if W:
                # W is a list of Warning instances
                for warning in W:
                    st.warning(f"Advertencia: {warning.message}")

    except Exception as e:
        st.error(f"Error procesando datos: {e}")


def show_reults(method, time_domain_results, frequency_domain_results):
    st.write("**Resultados temporales**")
    for i in time_domain_results.keys():
        value = time_domain_results[i]
        if hasattr(value, 'values'):
            value = value.values[0]
        st.markdown(f"**{i}**: {value}\n")

    st.write("**Resultados en Frecuencia**")
    for i in frequency_domain_results.keys():
        value = frequency_domain_results[i]
        if hasattr(value, 'values'):
            value = value.values[0]
        st.markdown(f"**{i}**: {value}\n")

    if method == "PYHRV":
        st.write(time_domain_results[20])
        st.write(frequency_domain_results[8])


def data_analysis(rr_intervals):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        method = "HRV-ANALYSIS"
        st.markdown(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_reults(method, time_domain_results, frequency_domain_results)
    with col2:
        method = "NEUROKIT2"
        st.markdown(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_reults(method, time_domain_results, frequency_domain_results)
    with col3:
        method = "PYHRV"
        st.markdown(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_reults(method, time_domain_results, frequency_domain_results)
    with col4:
        method = "HEARTPY"
        st.markdown(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_reults(method, time_domain_results, frequency_domain_results)





def hrv_comp():
    # Streamlit app setup
    st.title("Análisis HRV 🫀")
    st.sidebar.markdown("# Análisis HRV 🫀")

    # Upload CSV file
    uploaded_file = st.file_uploader("Cargar archivo CSV con data (bpm)", type="csv")

    if uploaded_file:
        # Read CSV file, skipping the first two rows
        # Read CSV with Polars
        data = pl.read_csv(
            uploaded_file,
            skip_rows=2,
            encoding="utf8-lossy",  # handles UTF-8 & BOM
        )
        # st.dataframe(data)

        # Display the raw data
        if st.sidebar.toggle("Mostrar datos"):
            st.write("### Raw Data")
            st.write(data)

        hrvalues, rr_intervals, ts, temp, showTemp = data_extraction(data)
        data_plot(hrvalues, rr_intervals, ts, temp, showTemp)

        data_analysis(rr_intervals)

    else:
        st.info("Subir archivo para realizar análisis")


def main():
    hrv_comp()


if __name__ == "__main__":
    main()
