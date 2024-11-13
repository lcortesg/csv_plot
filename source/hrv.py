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
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, plot_psd, plot_poincare
import seaborn as sns
sns.set_style("darkgrid")
plt.style.use('ggplot')

def hrv():
    # Streamlit app setup
    st.title("Análisis HRV")

    # Upload CSV file
    uploaded_file = st.file_uploader("Cargar archivo CSV con data (bpm)", type="csv")

    if uploaded_file:
        # Read CSV file, skipping the first two rows
        data = pd.read_csv(uploaded_file)
        f2r = data.head(1)
        #st.write(f2r.head())

        if st.toggle("Mostrar información"):
            st.write("### Información")
            for i in f2r.keys():
                # Check if the value is a number and not NaN
                value = f2r[i].values[0]

                # Skip iteration if the value is NaN
                if (isinstance(value, (int, float)) and not math.isnan(value)) or isinstance(value, (str)):
                    st.markdown(f"""
                        **{i}**: {value}\n
                    """)


        new_column_names = data.iloc[1]
        #st.write(new_column_names)

        # Skip the first row and set the new column names
        data = data[2:]  # Skip the first row
        data.columns = new_column_names  # Set new column names

        #data = pd.read_csv(uploaded_file, skiprows=2)

        # Display the raw data
        if st.toggle("Mostrar datos"):
            st.write("### Raw Data")
            st.write(data)


        # Assuming a column named 'HR (bpm)' in the uploaded CSV file
        if 'HR (bpm)' in data.columns:

            heart_rate_bpm = data['HR (bpm)'].values
            temp = data['Temperatures (C)'].values

            st.write("### BPM & Temp")
            showTemp = st.toggle("¿Mostrar temperaturas?")
            # Create a Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=heart_rate_bpm, mode='lines+markers', name="BPM"))
            if showTemp: fig.add_trace(go.Scatter(y=temp, mode='lines+markers', name="Temp"))
            st.plotly_chart(fig)

            hrvalues = [int(x) for x in heart_rate_bpm]
            # Convert heart rate from bpm to R-R intervals in milliseconds
            rr_intervals = [60000 / x for x in hrvalues]
            #rr_intervals = 60000 / hrvalues  # in ms

            st.write("### Intervalos RR")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rr_intervals, mode='lines+markers', name="RR"))
            st.plotly_chart(fig)

            method = st.selectbox(
                "Seleccionar método de análisis",
                ("PYHRV", "HRV-ANALYSIS", "HEARTPY"),
            )


            try:
                with warnings.catch_warnings(record=True) as W:
                    # Time-domain HRV metrics
                    if method == "PYHRV":
                        url = "https://pyhrv.readthedocs.io/en/latest/"
                        time_domain_results = td.time_domain(rr_intervals)
                        frequency_domain_results = fd.welch_psd(rr_intervals, show=False)
                    if method == "HRV-ANALYSIS":
                        url = "https://aura-healthcare.github.io/hrv-analysis/"
                        time_domain_results = get_time_domain_features(rr_intervals)
                        frequency_domain_results = get_frequency_domain_features(rr_intervals)
                    if method == "HEARTPY":
                        url = "https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/"
                        working_data, measures = hp.process_rr(rr_intervals)
                        time_domain_results = measures
                        frequency_domain_results =  get_frequency_domain_features(rr_intervals)#hp.hrv(working_data, sample_rate=1.0)
                    st.markdown(f"Consultar más información sobre la librería utilizada en el siguiente link: [{method}](%s)" % url)
                    if W:
                            # W is a list of Warning instances
                            for warning in W:
                                st.warning(f"Advertencia: {warning.message}")



                if st.toggle("Resultados temporales"):
                    if method == "PYHRV": st.write(time_domain_results[20])
                    for i in time_domain_results.keys():
                        st.markdown(f"""
                            **{i}**: {time_domain_results[i]}\n
                            """)
                if method != "HEARTPY":
                    if st.toggle("Resultados en Frecuencia"):
                        if method == "PYHRV": st.write(frequency_domain_results[8])
                        for i in frequency_domain_results.keys():
                            st.markdown(f"""
                                **{i}**: {frequency_domain_results[i]}\n
                                """)

            except Exception as e:
                st.error(f"Error procesando datos: {e}")

        else:
            st.error("El archivo CSV debe contener la columna 'HR (bpm)'.")
    else:
        st.info("Subir archivo para realizar análisis")
