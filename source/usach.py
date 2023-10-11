# -*- coding: utf-8 -*-
"""
@file    : CSV Converter
@brief   : Handles TXT to CSV file conversion.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import io
import csv
import math
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from scipy import signal
from scipy.signal import butter, filtfilt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def merge_qtm():
    st.markdown("# Comparación QTM/ABMA ❄️")
    st.sidebar.markdown("# Comparación QTM/ABMA ❄️")
    st.markdown("## Datos QTM")
    uploaded_files = st.file_uploader(
        "Hola Estudiante! elige los archivos TXT para convertir",
        type=["txt"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos TXT para convertir",
        key="txt_files",
    )

    perrito = {}
    tortuguita = {}
    cont = 0
    if len(uploaded_files) > 0:
        plot_data = st.checkbox(f'¿Graficar datos?')
        for uploaded_file in uploaded_files:
            data = uploaded_file.read()
            name = uploaded_file.name
            try:
                p0 = name.split(".")[0].split("_")[0]
                p1 = name.split(".")[0].split("_")[1]
                p2 = name.split(".")[0].split("_")[2]
            except:
                p0 = name.split(".")[0].split("-")[0]
                p1 = name.split(".")[0].split("-")[1]
                p2 = name.split(".")[0].split("-")[2]
            cast = p0
            side = p1 if len(p1) == 3 else p2
            part = p2 if len(p1) == 3 else p1
            
            file = str(data, "utf-8").split("\n")
            file.pop(0)
            filetxt = ""
            gatito = True
            for line in file:
                line = line.split()[-4:]
                if gatito:
                    line[1] = f"{part}_X"
                    line[2] = f"{part}_Y"
                    line[3] = f"{part}_Z"
                    gatito = False
                filetxt = filetxt + ",".join(line) + "\n"

            format = name[-3:]
            if plot_data:
                st.write(f"### {part}")
                st.download_button(
                    label=f'Descargar {name.replace(format,"csv")}',
                    data=filetxt,
                    file_name=name.replace(format, "csv"),
                    mime="text/csv",
                    key=cont,
                )
            cont = cont + 1

            
            buffer = io.StringIO(filetxt)
            df = pd.read_csv(filepath_or_buffer=buffer)
            keys = df.keys()
            for key in keys:
                perrito[key] = df[key]
                if "X" in key or "Frame" in key:
                    tortuguita[key.split("_")[0].lower()] = df[key]

            df = df.set_index("Frame")
            if plot_data:
                st.line_chart(df)
        pajarito = pd.DataFrame(perrito)
        hamster = pd.DataFrame(tortuguita)
        hamster = hamster.set_index("frame")
        # pajarito = pajarito.set_index('Frame')
        csv = convert_df(pajarito)
        st.download_button(
            "Descargar CSV", csv, f"{cast}-{side}.csv", "text/csv", key="download-csv"
        )
        return True, hamster, side
        
    else:
        return False, False, False


def load_abma(sideq):
    
    side = "LI" if sideq == "izq" else "LD"
    st.markdown("## Datos ABMA")
    uploaded_file = st.file_uploader(
        "Hola Estudiante! elige el archivo CSV para comparar",
        type=["csv"],
        accept_multiple_files=False,
        help="Selecciona un archivo csv para comparar",
        key="csv_files",
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        keys = df.keys()
        rinoceronte = {}
        parts = ["cadera", "rodilla", "tobillo", "frame"]
        for key in keys:
            if key.split()[0].lower() in parts:
                if side == key.split()[-1] or "frame" in key:
                    rinoceronte[key.split()[0].lower()] =  df[key]
        dfa = pd.DataFrame(rinoceronte)
        dfa = dfa.dropna()
        dfa = dfa.set_index("frame")
        return True, dfa
    else:
        return False, False
   

def plot(dfq, dfa):
    st.write(f"## QTM")
    st.line_chart(dfq)
    st.write(f"## ABMA")
    inv_hip = st.checkbox(f'¿Invertir Cadera?')
    inv_knee = st.checkbox(f'¿Invertir Rodilla?')
    inv_ankle = st.checkbox(f'¿invertir Tobillo?')
    dfan = {}
    values = st.slider(
        'Select a range of values',
        int(dfa.index[0]), int(dfa.index[-1]), (int(dfa.index[0]), int(dfa.index[-1])))
    dfan["cadera"] = dfa["cadera"][values[0]:values[1]]
    dfan["rodilla"] = dfa["rodilla"][values[0]:values[1]]
    dfan["tobillo"] = dfa["tobillo"][values[0]:values[1]]
    dfan["frame"] = dfa.index[values[0]-int(dfa.index[0]):values[1]-int(dfa.index[0]-1)]

    if inv_hip:
        dfan["cadera"] = -dfan["cadera"]
    if inv_knee:
        dfan["rodilla"] = -dfan["rodilla"]
    if inv_ankle:
        dfan["tobillo"] = -dfan["tobillo"]
    dfan = pd.DataFrame(dfan)
    dfan = dfan.set_index("frame")
    st.line_chart(dfan)
    
    return dfan


def butter_lowpass_filter(data, cutoff, fs, order):
    if order == 0:
        return data
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalise frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)  # Filter data
    return y


def compare(dfq, dfa):
    inverse = False
    samp = st.selectbox("Selecciona la frecuencia de muestreo de QTM", (100, 120))
    cutoff = st.slider("Seleccionar la frecuencia de corte", 4, 10, 6)
    order = st.slider("Selecciona el orden del filtro", 1, 8, 1)

    parts = ["cadera", "rodilla", "tobillo"]

    for part in parts:
        st.write(f"### {part}")
        abma_raw = dfa[part].to_list()
        abma = butter_lowpass_filter(data=abma_raw, cutoff=cutoff, fs=120, order=order)
        qtm = dfq[part].to_list()
        if samp != 120:
            qtm = np.interp(
                np.arange(0, len(qtm), samp / 120), np.arange(0, len(qtm)), qtm
            )
        number = st.number_input(f'Inserte el desfase de {part}', min_value=-90, max_value=90, value=0, step=1, key=part)
        abmac = [x - number for x in abma]
        # st.write(f'len(qtm): {len(qtm)}')
        # st.write(f'len(abma): {len(abma)}')
        # if len(qtm) > len(abma):

        shft = np.argmax(signal.correlate(qtm, abmac)) - len(abmac)
        if shft < 0:
            inverse = True
            shft = np.argmax(signal.correlate(abmac, qtm)) - len(qtm)
        # st.write(f"Shift: {shft}")
        if not inverse:
            # st.write("Not Inversed")
            if shft + len(dfa[part]) < len(qtm):
                qtmc = qtm[shft : shft + len(dfa[part])]
            else:
                qtmc = qtm[shft : len(qtm)]
                abmac = abmac[0 : len(qtmc)]
        if inverse:
            # st.write("Inversed")
            if shft + len(dfa[part]) < len(qtm):
                qtmc = qtm[shft : len(qtm)]
                abmac = abmac[0 : len(qtmc)]
                
            else:
                abmac = abmac[shft : shft + len(dfa[part])]
                qtmc = qtm[0: len(abmac)]
        
        # st.write(f'len(qtmc): {len(qtmc)}')
        # st.write(f'len(abmac): {len(abmac)}')
        # if len(abma) > len(qtm):
        #     shft = np.argmax(signal.correlate(abmac, qtm)) - len(qtm)  
        #     qtmc = qtm
        #     abmac = abmac[shft : shft + len(dfa[part])]
        
        
        
        errabs = np.absolute(np.subtract(qtmc,abmac))
        MSE = np.square(np.subtract(qtmc,abmac)).mean() 
        rmse = math.sqrt(MSE)

        a = qtmc
        b = abmac
        # a = (a - np.mean(a)) / (np.std(a) * len(a))
        # b = (b - np.mean(b)) / (np.std(b))
        # c = np.correlate(a, b, 'full')
        # a = a / np.linalg.norm(a)
        # b = b / np.linalg.norm(b)
        # c = np.correlate(a, b, mode = 'full')
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        c = np.correlate(a, b, 'full')
        distance = dtw.distance(qtmc, abmac)

        dft = {
            f"QTM - {part}": qtmc,
            f"ABMA - {part}": abmac,
            # f"ERROR ABS - {part}": errabs,
            # f"RMSE - {part}": rmse,
            # f"ABMA_RAW - {part}": abma_raw,
        }
        errores ={
            "Error máximo absoluto": max(errabs),
            "Error mínimo absoluto": min(errabs),
            "Error medio absoluto": errabs.mean(),
            "Error cuadrático medio": rmse,
            "Correlación cruzada máxima": max(c),
            "Distancia promedio DTW": distance/len(abmac),
        }

        dft = pd.DataFrame(dft)
        st.markdown(f"##### Gráficos de las señales")
        st.line_chart(dft)
        
        st.markdown(f"##### Correlación cruzada")
        st.line_chart(c)

        path = dtw.warping_path(qtmc, abmac)
        dtwvis.plot_warping(qtmc, abmac, path, filename="warp.png")
        st.markdown(f"##### DTW warping path")
        st.image("warp.png", use_column_width=True)

        st.markdown(f"##### Tabla de resultados")
        st.write(errores)


def usach_plot():
    merged, dfq, sideq = merge_qtm()
    if merged:
        loaded, dfa = load_abma(sideq)
        if loaded:
            dfan = plot(dfq, dfa)
            compare(dfq, dfan)
