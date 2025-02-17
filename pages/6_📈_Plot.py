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
from PIL import Image
import plotly.graph_objects as go
im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)

def csv_plot():

    
    st.markdown("# CSV Plot 📈")
    st.sidebar.markdown("# CSV Plot 📈")

    uploaded_file = st.file_uploader(
        "Elige el archivo CSV para graficar",
        type=["csv"],
        accept_multiple_files=False,
        help="Selecciona un archivo CSV para graficar",
    )
        
    if uploaded_file:
        filename = uploaded_file.name
        dataframe = pd.read_csv(uploaded_file)
        #st.write(dataframe)
        cols = []
        for col in dataframe:
            cols.append(col)
        #st.write(cols)

        col = st.selectbox(
            "Seleccionar variable",
            cols,
        )

        min_floor = st.number_input("Piso de ruido", 0, 3000, 100, 100)
        percentile = st.number_input("Percentil", 95.0, 100.0, 100.0, 0.1)
        quantile = percentile / 100

        fuerza = dataframe[col].tolist()
        fuerza_nz = []
        for value in fuerza:
            if abs(value) >= min_floor:
                fuerza_nz.append(value)
        
        data_max =  np.round(np.max(np.percentile(fuerza, percentile)), 1)
        data_mean = np.round(np.mean(fuerza_nz), 1)
        data_std = np.round(np.std(fuerza_nz), 1)
        data_var = np.round(np.var(fuerza_nz), 1)
        data_med = np.round(np.median(fuerza_nz), 1)

        vars = {
            f"Máximo": data_max,
            f"Valor Medio": data_mean,
            f"Desviación Estándar": data_std,
            #f"Varianza": data_var,
            f"Mediana": data_med,
        }

        variables = {
            f"Máximo": len(fuerza)*[data_max],
            f"Valor Medio": len(fuerza)*[data_mean],
            f"Desviación Estándar": len(fuerza)*[data_std],
            #f"Varianza": len(fuerza)*[data_var],
            f"Mediana": len(fuerza)*[data_med],
        }

        all_options = ["Máximo", "Valor Medio", "Desviación Estándar", "Mediana"]

        options = st.multiselect(
            "Estadísticas",
            all_options,
            ["Máximo"],
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=fuerza, mode='lines', name=col))
        for option in options:
            fig.add_trace(go.Scatter(y=variables[option], mode='lines', name=option))
        st.plotly_chart(fig)
        st.write(vars)
        return True

    return False


def main():
    csv_plot()

if __name__ == "__main__":
    main()
