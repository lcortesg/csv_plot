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

    st.title("CSV Plot")

    uploaded_file = st.file_uploader("Hola Nico! elige un archivo CSV")
    if uploaded_file is not None:
        if uploaded_file.name.split(".")[-1] == "csv":
            dataframe = pd.read_csv(uploaded_file)
            force_plot(dataframe)
        else:
            st.write("Nico! El archivo tiene que ser un CSV! >:(")


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


def force_plot(dataframe):
    dataframe.rename(columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True)
    data = dataframe["Fuerza"]

    data_max = [np.max(data)] * len(data)
    data_avg = [np.average(data)] * len(data)
    data_std = [np.std(data)] * len(data)

    options = st.multiselect(
        "¿Que quieres graficar?",
        ["Máximo", "Valor Medio", "Desviación Estándar"],
        ["Máximo", "Valor Medio"],
    )

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
    st.line_chart(df)

    data_aux = {
        "Máximo": trunc(data_max[0], 1),
        "Valor Medio": trunc(data_avg[0], 1),
        "Desviación Estándar": trunc(data_std[0], 1),
    }
    st.write(data_aux)


if __name__ == "__main__":
    main()
