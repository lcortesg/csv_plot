# -*- coding: utf-8 -*-
"""
@file     : CSV Handler
@brief   : Handles CSV and TXT file conversion and plotting.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from io import StringIO


def main():
    im = Image.open("assets/logos/favicon.png")
    st.set_page_config(
        page_title="CSV Handler",
        page_icon=im,
        layout="wide",
    )
    page_names_to_funcs = {
        "Principal": main_page,
        "Plot": csv_plot,
        "Split": csv_split,
        "Merge": csv_merge,
        "Convert": csv_convert,
    }
    selected_page = st.sidebar.selectbox("Hola Nicco! ¿Que quieres hacer hoy?", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


def main_page():
    st.markdown("# Página principal ✨")
    st.sidebar.markdown("# Página Principal ✨")


@st.cache
def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def csv_merge():
    st.markdown("# CSV Merge 🪢")
    st.sidebar.markdown("# CSV Merge 🪢")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos CSV para mezclar",
        type=['csv'],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para mezclar",
    )

    if len(uploaded_files) > 1:
        
        df = pd.concat(map(pd.read_csv, uploaded_files), ignore_index=True)
        csv_data = df.to_csv()
        csv_name = ""
        for uploaded_file in uploaded_files:
            csv_name = csv_name + "+" + uploaded_file.name.split(".")[0]
        csv_name = csv_name[1:-1]
        st.subheader(csv_name + ".csv")
        st.download_button(
            label=f"Descargar CSV",
            data=csv_data,
            file_name=csv_name + ".csv",
            mime="text/csv",
        )
    return 0


def csv_split(length=3138):
    st.markdown("# CSV Split ⚡️")
    st.sidebar.markdown("# CSV Split ⚡️")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos CSV para dividir",
        type=['csv'],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para dividir",
    )

    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            dataframe = pd.read_csv(uploaded_file)
            parts = int(len(dataframe["0"]) / length)
            st.subheader(filename)
            for i in range(parts):
                start = length * i + i
                stop = length * (i + 1)
                data = dataframe["0"][start:stop]
                df = pd.DataFrame(data)
                csv_data = df.to_csv()
                csv_name = f'{filename.split(".")[0]}_{i+1}'
                st.download_button(
                    label=f"Descargar CSV parte {i+1}",
                    data=csv_data,
                    file_name=csv_name + ".csv",
                    mime="text/csv",
                )
    return 0


def csv_plot():
    st.markdown("# CSV Plot 📈")
    st.sidebar.markdown("# CSV Plot 📈")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos CSV para graficar",
        type=['csv'],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para graficar",
    )

    if len(uploaded_files) > 0:
        options = st.multiselect(
            "¿Que quieres graficar?",
            ["Máximo", "Valor Medio", "Desviación Estándar"],
            ["Máximo", "Valor Medio"],
        )
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            dataframe = pd.read_csv(uploaded_file)
            dataframe.rename(
                columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True
            )
            data = dataframe["Fuerza"]
            data_nz = []
            for value in data: 
                if abs(value) >= 1: data_nz.append(value)
            
            data_max = [np.max(data.quantile(0.99))] * len(data)
            data_avg = [np.average(data_nz)] * len(data)
            data_std = [np.std(data_nz)] * len(data)

            variables = {
                "Fuerza": data,
                f"Máximo: {trunc(data_max[0],1)}": data_max,
                f"Valor Medio: {trunc(data_avg[0],1)}": data_avg,
                f"STD: {trunc(data_std[0],1)}": data_std,
            }

            if "Máximo" not in options:
                del variables[f"Máximo: {trunc(data_max[0],1)}"]
            if "Valor Medio" not in options:
                del variables[f"Valor Medio: {trunc(data_avg[0],1)}"]
            if "Desviación Estándar" not in options:
                del variables[f"STD: {trunc(data_std[0],1)}"]

            df = pd.DataFrame(variables)
            st.subheader(filename)
            st.line_chart(df)

            data_aux = {
                "Máximo": trunc(data_max[0], 1),
                "Valor Medio": trunc(data_avg[0], 1),
                "Desviación Estándar": trunc(data_std[0], 1),
            }

            st.write(data_aux)
    return 0

def csv_convert():
    st.markdown("# TXT Convert ❄️")
    st.sidebar.markdown("# TXT Convert ❄️")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos TXT para convertir",
        type=['txt'],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos TXT para convertir",
    )

    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            fileread = uploaded_file.read()
            filename = uploaded_file.name
            new_line = ""
            file = str(fileread,"utf-8").split("\n")
            word_list = file[1].split("\t")
            word_list[0] = "XY"
            new_line = ",".join(word_list)
            file.pop(0)
            file[0] = new_line
            filetxt = ""
            for line in file:
                filetxt = filetxt+line.replace("\t",",")

            st.download_button(
                label=f'Descargar {filename.replace("TXT","csv")}',
                data=filetxt,
                file_name=filename.replace("TXT","csv"),
                mime="text/csv",
                )
    return 0



if __name__ == "__main__":
    main()