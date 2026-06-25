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
    """Replace zeros with NaN and display header info if requested."""
    data.replace(0, np.nan, inplace=True)
    if st.sidebar.toggle("Mostrar información"):
        st.write("### Información")
        f2r = data.head(1)
        for i in f2r.keys():
            value = f2r[i].values[0]
            if (isinstance(value, (int, float)) and not math.isnan(value)) or isinstance(value, (str)):
                st.markdown(f"**{i}**: {value}\n")

    if "HR" not in data.columns:
        new_column_names = data.iloc[1]
        data = data[2:]
        data.columns = new_column_names
        data.rename(columns={"HR (BPM)": "HR"}, inplace=True)
    return data


def identify_columns(data):
    """Locate HR, timestamp, and temperature columns by name patterns."""
    hr_column = None
    time_column = None
    temp_column = None
    for col in data.columns:
        if "HR" in col or "BPM" in col:
            hr_column = col
        if "Time" in col or "Timestamp" in col:
            time_column = col
        if "Temperatures" in col:
            temp_column = col
    return hr_column, time_column, temp_column


def data_extraction(data):
    """Extract HR values, RR intervals, timestamps, and temps. Return lists."""
    hr_column, time_column, temp_column = identify_columns(data)

    if hr_column not in data.columns:
        st.error("❌ No HR/BPM column found. Check column names.")
        return [], [], [], [], False

    hrvalues = data[hr_column]
    hrvalues = [int(x) for x in hrvalues if not math.isnan(x)]
    rr_intervals = [60000 / x for x in hrvalues]

    ts = list(range(len(hrvalues)))
    if time_column and time_column in data.columns:
        ts = data[time_column]

    showTemp = False
    temp = []
    if temp_column and temp_column in data.columns:
        temp = data[temp_column]
        showTemp = st.sidebar.toggle("¿Mostrar temperaturas?")

    return hrvalues, rr_intervals, ts, temp, showTemp



def data_plot(hrvalues, rr_intervals, ts, temp, showTemp):
    """Plot BPM and RR intervals. Include temperature if showTemp=True."""
    st.write("### BPM & Temp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=hrvalues, mode="lines+markers", name="BPM"))
    fig.update_layout(
        title="Pulse Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Pulse (BPM)",
        template="plotly_white",
    )
    if showTemp:
        fig.add_trace(go.Scatter(x=ts, y=temp, mode="lines+markers", name="Temp"))
    st.plotly_chart(fig)

    st.write("### Intervalos RR")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=rr_intervals, mode="lines+markers", name="RR"))
    fig.update_layout(
        title="RR Intervals Over Time",
        xaxis_title="Time (s)",
        yaxis_title="RR Interval (ms)",
        template="plotly_white",
    )
    st.plotly_chart(fig)



def get_results(method, rr_intervals):
    """Compute time and frequency domain HRV metrics using selected method."""
    try:
        with warnings.catch_warnings(record=True) as W:
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
                _, measures = hp.process_rr(rr_intervals)
                time_domain_results = measures
                frequency_domain_results = get_frequency_domain_features(rr_intervals)
                return time_domain_results, frequency_domain_results
            if method == "NEUROKIT2":
                url = "https://neuropsychology.github.io/NeuroKit/"
                peaks = nk.intervals_to_peaks(rr_intervals)
                time_domain_results = nk.hrv_time(peaks)
                frequency_domain_results = nk.hrv_frequency(peaks)
                return time_domain_results, frequency_domain_results
            st.markdown(
                f"Consultar más información sobre la librería utilizada en el siguiente link: [{method}]({url})"
            )
            if W:
                # W is a list of Warning instances
                for warning in W:
                    st.warning(f"Advertencia: {warning.message}")

    except Exception as e:
        st.error(f"Error procesando datos: {e}")


def show_results(method, time_domain_results, frequency_domain_results):
    """Display time and frequency domain results. PYHRV shows extra indices."""
    docs = {
        "PYHRV": "https://pyhrv.readthedocs.io/en/latest/",
        "HRV-ANALYSIS": "https://aura-healthcare.github.io/hrv-analysis/",
        "HEARTPY": "https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/",
        "NEUROKIT2": "https://neuropsychology.github.io/NeuroKit/",
    }
    st.info(f"[📖 Docs]({docs.get(method, '#')})")

    for title, results in [("Resultados temporales", time_domain_results), ("Resultados en Frecuencia", frequency_domain_results)]:
        st.warning(f"**{title}**")
        for i in results.keys():
            value = results[i]
            if hasattr(value, 'values'):
                value = value.values[0]
            st.markdown(f"**{i}**: {value}\n")

    if method == "PYHRV":
        st.write(time_domain_results[20])
        st.write(frequency_domain_results[8])


def data_analysis(rr_intervals):
    """Run HRV analysis with 4 methods in parallel columns."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        method = "HRV-ANALYSIS"
        st.success(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_results(method, time_domain_results, frequency_domain_results)
    with col2:
        method = "NEUROKIT2"
        st.success(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_results(method, time_domain_results, frequency_domain_results)
    with col3:
        method = "PYHRV"
        st.success(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_results(method, time_domain_results, frequency_domain_results)
    with col4:
        method = "HEARTPY"
        st.success(f"**{method}**")
        time_domain_results, frequency_domain_results = get_results(method, rr_intervals)
        show_results(method, time_domain_results, frequency_domain_results)





def hrv_comp():
    """Main HRV analysis app. Upload CSV, extract data, plot, compute metrics."""
    st.title("Análisis HRV 🫀")
    st.sidebar.markdown("# Análisis HRV 🫀")


    uploaded_file = st.file_uploader("Cargar archivo CSV con data (bpm)", type="csv")

    if uploaded_file:
        try:
            data = pl.read_csv(uploaded_file, skip_rows=2, encoding="utf8-lossy")
        except Exception as e:
            st.error(f"❌ Error leyendo archivo: {e}")
            return

        if st.sidebar.toggle("Mostrar datos"):
            st.write("### Raw Data")
            st.write(data)

        hrvalues, rr_intervals, ts, temp, showTemp = data_extraction(data)

        if not hrvalues:
            st.warning("⚠️ No hay datos válidos para analizar.")
            return

        data_plot(hrvalues, rr_intervals, ts, temp, showTemp)
        data_analysis(rr_intervals)
    else:
        st.info("📁 Subir archivo CSV para realizar análisis")


def main():
    hrv_comp()


if __name__ == "__main__":
    main()
