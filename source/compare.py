import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
import warnings


def compare():
    # Streamlit app setup
    st.title("Análisis Facial")

    # Upload CSV file

    uploaded_files = st.file_uploader(
        "Elige el archivo CSV para comparar",
        type=["csv", "tsv"],
        accept_multiple_files=True,
        help="Selecciona un archivo csv para comparar",
        key="csv_files",
    )
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            if "csv" in uploaded_file.name:
                # Read data
                data1 = pd.read_csv(uploaded_file)
                # Read head
                column_names = data1.dropna(axis=1, how='all').columns.tolist()
                # Trim columns
                col_trim = []
                for col in column_names:
                    if "eye" in col.lower() or "iris" in col.lower():
                        col_trim.append(col)

                opt1 = st.selectbox(
                    "Seleccionar variable CSV",
                    (col_trim),
                    index=0,
                    placeholder="None",
                )
                # Plot
                fig = px.line(data1, x="frame", y=opt1, title=f'{opt1}')
                st.write(fig)

            if "tsv" in uploaded_file.name:
                # Read data
                data2 = pd.read_csv(uploaded_file, sep='\t')

                #Filter by sensor
                sensor_value = st.selectbox(
                    "Seleccionar sensor",
                    (data2["Sensor"].dropna().unique()),
                    placeholder="Eye Tracker"
                )
                data2 = data2[data2['Sensor'] == sensor_value]

                # Filter by participant
                participant_value = st.selectbox(
                    "Seleccionar participante",
                    (data2["Participant name"].unique()),
                    placeholder="None"
                )
                data2 = data2[data2['Participant name'] == participant_value]

                # Read head
                column_names = data2.dropna(axis=1, how='all').columns.tolist()
                col_trim = column_names
                opt2 = st.selectbox(
                    "Seleccionar variable TSV",
                    (col_trim),
                    index=0,
                    placeholder="None",
                )

                # Plot
                fig = px.line(data2, x=data2.index, y=opt2, title=f'{opt2}')
                st.write(fig)



    else:
        st.info("Subir archivo para realizar análisis")
