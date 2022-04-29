import streamlit as st

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import json

# import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from scipy.signal import butter, filtfilt
from numpy import diff


def main():
    im = Image.open("assets/logos/favicon.png")
    st.set_page_config(
        page_title="Lanek Plot",
        page_icon=im,
        layout="wide",
    )

    st.title("Lanek Plot")

    uploaded_file = st.file_uploader("Hola Nico! elige un archivo CSV")
    if uploaded_file is not None:
        # st.write('csv' in uploaded_file.name.split('.'))
        if uploaded_file.name.split(".")[-1] == "csv":
            dataframe = pd.read_csv(uploaded_file)
            force_plot(dataframe)
        else:
            st.write("Nico! El archivo tiene que ser un CSV!")


@st.cache
def butter_lowpass_filter(data, cutoff=6, fs=120, order=8):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalise frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)  # Filter data
    return y


@st.cache
def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)

@st.cache
def get_max(data):
    data_max = []
    for i in range(len(data)):
        data_max.append(np.max(data))
    return data_max

@st.cache
def get_avg(data):
    data_avg = []
    for j in range(len(data)):
        data_avg.append(np.average(data))
    return data_avg

@st.cache
def get_std(data):
    data_std = []
    for k in range(len(data)):
        data_std.append(np.std(data))
    return data_std

def force_plot(dataframe):
    dataframe.rename(columns={"Unnamed: 0": "Tiempo", "0": "Fuerza"}, inplace=True)
    data = dataframe["Fuerza"]

    data_max = get_max(data)
    data_avg = get_avg(data)
    data_std = get_std(data)

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
    # data_filt = butter_lowpass_filter(data, cutoff=6, fs=120, order=8)
    st.line_chart(df)

    data_aux = {
        "Máximo": trunc(data_max[0],1),
        "Valor Medio": trunc(data_avg[0],1),
        "Desviación Estándar": trunc(data_std[0],1),
    }
    st.write(data_aux)


if __name__ == "__main__":
    main()
