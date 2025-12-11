# -*- coding: utf-8 -*-
"""
@file    : 3_🫀_HRV.py
@brief   : Handles HRV processing.
@date    : 2024/08/22
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import streamlit as st
import pandas as pd
import heartpy as hp
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import math
import warnings
import polars as pl
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, plot_psd, plot_poincare
from PIL import Image
im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)


def data_cleaning(data, name):
    data.replace(0, np.nan, inplace=True) 
    # if st.toggle("Mostrar información", key=f"{name}-info"):
    #     st.write("### Información")
    #     f2r = data.head(1)
    #     for i in f2r.keys():
    #         # Check if the value is a number and not NaN
    #         value = f2r[i].values[0]
    #         # Skip iteration if the value is NaN
    #         if (isinstance(value, (int, float)) and not math.isnan(value)) or isinstance(value, (str)):
    #             st.markdown(f"""
    #                 **{i}**: {value}\n
    #             """)

    if "HR" not in data.columns:
        new_column_names = data.iloc[1]
        # Skip the first row and set the new column names
        data = data[2:]  # Skip the first row
        data.columns = new_column_names  # Set new column names
        #data = pd.read_csv(uploaded_file, skiprows=2)
        data.rename(columns={"HR (BPM)": "HR"}, inplace=True)
    return data


def identify_columns(data):
    hr_column = None
    time_column = None
    for col in data.columns:
        if 'HR' in col or 'BPM' in col:
            hr_column = col
        if 'Time' in col or 'Timestamp' in col:
            time_column = col
        if "Temperatures" in col:
            temp_column = col   
    return hr_column, time_column, temp_column

def data_extraction(data):
    hr_column, time_column, temp_column = identify_columns(data)
    if hr_column in data.columns:
        hrvalues = data[hr_column]#.values
        hrvalues = [int(x) for x in hrvalues if not math.isnan(x)]
        rr_intervals = [60000 / x for x in hrvalues]
    ts = list(range(len(hrvalues)))
    if time_column in data.columns:
        ts = data[time_column]#.values
    showTemp = False
    temp = []
    if temp_column in data.columns:
        temp = data[temp_column]#.values
        #showTemp = st.sidebar.toggle("¿Mostrar temperaturas?")
    return hrvalues, rr_intervals, ts, temp, showTemp


def data_plot(hrvalues, rr_intervals, ts, temp, showTemp, name):
    st.write("### BPM & Temp")
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=hrvalues, mode='lines+markers', name="BPM"))
    fig.update_layout(
        title="Pulse Over Time",
        xaxis_title="Time (s)",  # Replace with appropriate unit for 'ts'
        yaxis_title="Pulse (BPM)",
        template="plotly_white",  # Optional: Set a clean background style
    )
    if showTemp:
        fig.add_trace(go.Scatter(x=ts, y=temp, mode='lines+markers', name="Temp"))
    st.plotly_chart(fig)

    st.write("### Intervalos RR")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=rr_intervals, mode='lines+markers', name="RR"))
    fig.update_layout(
        title="RR Intervals Over Time",
        xaxis_title="Time (s)",  # Replace with appropriate unit for 'ts'
        yaxis_title="RR Interval (ms)",
        template="plotly_white",  # Optional: Set a clean background style
    )
    st.plotly_chart(fig)

def data_analysis(rr_intervals, name, temps, freqs):
    methods = ["PYHRV", "HEARTPY", "HRV-ANALYSIS"]
    variables = ["sdnn", "rmssd", "pnni_50", "pnn50"]
    results = {}
    results["FILENAME"] = name
    time_domain_results = {}
    frequency_domain_results = {}

    for method in methods:
        try:
            if method == "PYHRV":
                time_domain_results = td.time_domain(rr_intervals)
            if method == "HRV-ANALYSIS":
                time_domain_results = get_time_domain_features(rr_intervals)
                frequency_domain_results = get_frequency_domain_features(rr_intervals)
            if method == "HEARTPY":
                working_data, measures = hp.process_rr(rr_intervals)
                time_domain_results = measures

            if temps or freqs:
                st.write(f"#### {method}")
            if temps:
                st.write("##### Temporal")
                for i in time_domain_results.keys():
                    st.markdown(f"""
                        **{i}**: {time_domain_results[i]}\n
                        """)
            if freqs:
                if method == "HRV-ANALYSIS":
                    st.write("##### Frecuencia")
                    for i in frequency_domain_results.keys():
                        st.markdown(f"""
                            **{i}**: {frequency_domain_results[i]}\n
                            """)
                        
            for variable in variables:
                if variable in time_domain_results.keys():
                    if variable == "pnni_50":
                        results[f"{method}-pnn50"] = time_domain_results[variable]
                    else:
                        results[f"{method}-{variable}"] = time_domain_results[variable]
            if method == "HRV-ANALYSIS":
                variable = "lf_hf_ratio"
                results[f"{method}-{variable}"] = frequency_domain_results[variable]


        except Exception as e:
            st.error(f"Error procesando datos: {e}")

    return results

def hrv_comp():
    # Streamlit app setup
    st.title("Análisis HRV 🫀")
    st.sidebar.markdown("# Análisis HRV 🫀")

    # Upload CSV file
    files = st.file_uploader("Cargar archivo CSV con data (bpm)", type="csv", accept_multiple_files=True)
    results = []
    if files:
        plots = st.sidebar.toggle("Mostrar gráficos")
        datos = st.sidebar.toggle("Mostrar datos")
        temps = st.sidebar.toggle("Resultados temporales")
        freqs = st.sidebar.toggle("Resultados en frecuencia")
        res = st.sidebar.toggle("Mostrar resultados")
        for file in files:
            name = file.name
            # Read CSV file, skipping the first two rows
            # data = pd.read_csv(file)
            # data = data_cleaning(data, name)

            data = pl.read_csv(
                file,
                skip_rows=2,
                encoding="utf8-lossy"  # handles UTF-8 & BOM
            )
            
            if datos or plots or temps or freqs or res:
                st.write(f"### {name}")
            if datos:
                st.dataframe(data)

            hrvalues, rr_intervals, ts, temp, showTemp = data_extraction(data)

            if plots:  
                data_plot(hrvalues, rr_intervals, ts, temp, showTemp, name)
            result = data_analysis(rr_intervals, name, temps, freqs)
            results.append(result) 
            if res:
                st.write(result)
            
    
    if results:
        results_df = pd.DataFrame(results)
        st.write("### Resultados")
        st.dataframe(results_df)
        csv = results_df.to_csv(index=False).encode('utf-8')

        # Add download button
        st.download_button(
            label="📥 Descargar como CSV",
            data=csv,
            file_name="resultados.csv",
            mime="text/csv"
        )
            
    else:
        st.info("Subir archivos para realizar análisis")

def main():
    hrv_comp()

if __name__ == "__main__":
    main()
