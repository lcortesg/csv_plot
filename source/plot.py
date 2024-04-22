# -*- coding: utf-8 -*-
"""
@file    : CSV Plotter
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
    st.markdown("# CSV Force Plot 📈")
    st.sidebar.markdown("# CSV Force Plot 📈")

    uploaded_files = st.file_uploader(
        "Elige los archivos CSV para graficar",
        type=["csv"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para graficar",
    )

    if len(uploaded_files) > 0:

        all_options = ["Máximo", "Valor Medio", "Desviación Estándar", "Mediana", "Varianza"]

        options = st.multiselect(
            "¿Que quieres graficar?",
            all_options,
            ["Máximo", "Valor Medio"],
        )

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name
            dataframe = pd.read_csv(uploaded_file)
            dataframe.rename(
                columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True
            )
            tiempo = dataframe["Tiempo"]
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
                "Tiempo": tiempo,
                "Fuerza": fuerza,
                f"Máximo": data_max,
                f"Valor Medio": data_mean,
                f"Desviación Estándar": data_std,
                f"Varianza": data_var,
                f"Mediana": data_med,
            }

            df = pd.DataFrame(variables)
            df = df.set_index('Tiempo')

            for option in all_options:
                if option not in options:
                    df = df.drop(f'{option}', axis=1)

            st.subheader(filename)
            st.line_chart(df)

            data_aux = variables
            del data_aux["Tiempo"]
            del data_aux["Fuerza"]
            st.write(data_aux)

        return True

    return False
