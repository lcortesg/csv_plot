# -*- coding: utf-8 -*-
"""
@file     : CSV Plotter
@brief   : Handles CSV file plotting.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import numpy as np
import pandas as pd
import streamlit as st


def csv_plot():

    min_floor = 5
    percentile = 99.8
    quantile = percentile / 100
    st.markdown("# CSV Plot 📈")
    st.sidebar.markdown("# CSV Plot 📈")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos CSV para graficar",
        type=["csv"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para graficar",
    )

    if len(uploaded_files) > 0:

        options = st.multiselect(
            "¿Que quieres graficar?",
            ["Máximo", "Valor Medio", "Desviación Estándar", "Mediana", "Varianza"],
            ["Máximo", "Valor Medio"],
        )

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name
            dataframe = pd.read_csv(uploaded_file)
            dataframe.rename(
                columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True
            )
            fuerza = dataframe["Fuerza"]
            fuerza_nz = []

            for value in fuerza:
                if abs(value) >= min_floor:
                    fuerza_nz.append(value)

            data_max = np.round(np.max(np.percentile(fuerza, percentile)), 1)
            data_mean = np.round(np.mean(fuerza_nz), 1)
            data_std = np.round(np.std(fuerza_nz), 1)
            data_var = np.round(np.var(fuerza_nz), 1)
            data_med = np.round(np.median(fuerza_nz), 1)

            variables = {
                "Fuerza": fuerza,
                f"Máximo: {data_max}": data_max,
                f"Valor Medio: {data_mean}": data_mean,
                f"STD: {data_std}": data_std,
                f"Varianza: {data_var}": data_var,
                f"Mediana: {data_med}": data_med,
            }

            if "Máximo" not in options:
                del variables[f"Máximo: {data_max}"]
            if "Valor Medio" not in options:
                del variables[f"Valor Medio: {data_mean}"]
            if "Desviación Estándar" not in options:
                del variables[f"STD: {data_std}"]
            if "Varianza" not in options:
                del variables[f"Varianza: {data_var}"]
            if "Mediana" not in options:
                del variables[f"Mediana: {data_med}"]

            df = pd.DataFrame(variables)
            st.subheader(filename)
            st.line_chart(df)

            data_aux = {
                f"Máximo:": data_max,
                f"Valor Medio:": data_mean,
                f"Desviación Estándar:": data_std,
                f"Varianza:": data_var,
                f"Mediana:": data_med,
            }

            st.write(data_aux)

        return True

    return False
