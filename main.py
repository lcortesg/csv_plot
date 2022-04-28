import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import json
#import seaborn as sns
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
        page_title="ABMA Plot",
        page_icon=im,
        layout="wide",
    )

    st.title('Lanek Plot')

    uploaded_file = st.file_uploader("Elige el archivo CSV")
    if uploaded_file is not None:
        # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)

        # To convert to a string based IO:
        #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #st.write(stringio)

        # To read file as string:
        #string_data = stringio.read()
        #st.write(string_data)

        dataframe = pd.read_csv(uploaded_file)
        dataframe.rename(columns = {'Unnamed: 0':'Tiempo','0':'Fuerza'}, inplace = True)
        data = dataframe["Fuerza"]
        data_max=[]
        for i in range(len(data)):
            data_max.append(np.max(data))
        data_avg=[]
        for j in range(len(data)):
            data_avg.append(np.average(data))
        df = pd.DataFrame({"Fuerza": data,
                            f'Máximo: {trunc(data_max[0],1)}': data_max,
                            f'Valor Medio: {trunc(data_avg[0],1)}': data_avg})
        #st.write(df)

    
        #data_filt = butter_lowpass_filter(data, cutoff=6, fs=120, order=8)
        st.line_chart(df)
    #report()


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
    return np.trunc(values*10**decs)/(10**decs)
        
if __name__ == "__main__":
    main()
