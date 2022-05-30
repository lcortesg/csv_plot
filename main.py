import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from scipy.signal import butter, filtfilt


def main():
    im = Image.open("assets/logos/favicon.png")
    st.set_page_config(
        page_title="CSV Plot",
        page_icon=im,
        layout="wide",
    )

    option = st.sidebar.selectbox(
        "Hola Nico! ¿Que quieres hacer?", ("Plot", "Split", "Merge", "Convert")
    )

    if option == "Plot":
        csv_plot()

    if option == "Split":
        csv_split()

    if option == "Merge":
        csv_merge()
    
    if option == "Convert":
        csv_convert()


@st.cache
def butter_lowpass_filter(data, cutoff=6, fs=120, order=8):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


@st.cache
def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def csv_all(uploaded_files):
    all_csv = False
    for uploaded_file in uploaded_files:
        if uploaded_file.name.split(".")[-1] == "csv":
            all_csv = True
        else:
            all_csv = False
            st.caption("Nico! Todos los archivos tienen que ser CSV! >:(")
    if all_csv:
        return True
    else:
        return False
    
def txt_all(uploaded_files):
    all_csv = False
    for uploaded_file in uploaded_files:
        if uploaded_file.name.split(".")[-1] == "txt":
            all_csv = True
        else:
            all_csv = False
            st.caption("Nico! Todos los archivos tienen que ser TXT! >:(")
    if all_csv:
        return True
    else:
        return False


def csv_merge():
    st.title("CSV Merge")

    uploaded_files = st.file_uploader(
        "Hola Nico! elige los archivos CSV para mezclar",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para mezclar",
    )

    if len(uploaded_files) > 1:
        if csv_all(uploaded_files):
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
    st.title("CSV Split")

    uploaded_files = st.file_uploader(
        "Hola Nico! elige los archivos CSV para dividir",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para dividir",
    )

    if len(uploaded_files) > 0:
        if csv_all(uploaded_files):
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
    st.title("CSV Plot")

    uploaded_files = st.file_uploader(
        "Hola Nico! elige los archivos CSV para graficar",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para graficar",
    )

    if len(uploaded_files) > 0:
        options = st.multiselect(
            "¿Que quieres graficar?",
            ["Máximo", "Valor Medio", "Desviación Estándar"],
            ["Máximo", "Valor Medio"],
        )
        if csv_all(uploaded_files):
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                dataframe = pd.read_csv(uploaded_file)
                dataframe.rename(
                    columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True
                )
                data = dataframe["Fuerza"]
                data_max = [np.max(data)] * len(data)
                data_avg = [np.average(data)] * len(data)
                data_std = [np.std(data)] * len(data)

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
    st.title("TXT Convert")

    uploaded_files = st.file_uploader(
        "Hola Nico! elige los archivos TXT para convertir",
        accept_multiple_files=True,
        help="Selecciona uno o más archivos TXT para convertir",
    )

    if len(uploaded_files) > 0:
        options = st.multiselect(
            "¿Que quieres graficar?",
            ["Máximo", "Valor Medio", "Desviación Estándar"],
            ["Máximo", "Valor Medio"],
        )
        
        if txt_all(uploaded_files):
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file = open(filename, "r")
                for line in file:
                    st.write(line) 
    return 0


if __name__ == "__main__":
    main()
