# -*- coding: utf-8 -*-
"""
@file    : CSV Converter
@brief   : Handles TXT to CSV file conversion.
@date    : 2024/04/22
@version : 2.0.1
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import io
import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from numpy import mean
from numpy import var
from math import sqrt
import streamlit as st
from io import BytesIO
import scipy.stats
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import lognorm
from scipy import signal
from scipy.signal import butter, filtfilt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import euclidean
import spm1d

"""
from fastdtw import fastdtw
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")
"""

translate = {
    "knee": "Rodilla",
    "hip": "Cadera",
    "ankle": "Tobillo",
    "trunk": "Tronco"
}

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

def randomize(x):
    y = []
    for i in x:
        y.append(i+random.uniform(-0.1,0.1))
    return np.array(y)

def randomizeM(x, n):
    y = []
    for i in range(n):
        y.append(x+random.uniform(-0.1,0.1))
    return np.array(y)

def merge_qtm():
    parts = []
    st.markdown("# Comparación QTM/ABMA 🏃‍♂️")
    st.sidebar.markdown("# Comparación QTM/ABMA 🏃‍♂️️")
    st.markdown("## Datos QTM")
    uploaded_files = st.file_uploader(
        "Elige los archivos TXT para convertir",
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
            side = name.split(".")[0].split("_")[0]
            part = translate[name.split(".")[0].split("_")[1]]
            parts.append(part)

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
                    tortuguita[key.split("_")[0]] = df[key]

            df = df.set_index("Frame")
            if plot_data:
                st.line_chart(df)
        pajarito = pd.DataFrame(perrito)
        hamster = pd.DataFrame(tortuguita)
        hamster = hamster.set_index("Frame")
        # pajarito = pajarito.set_index('frame')
        csv = convert_df(pajarito)
        st.download_button(
            "Descargar CSV", csv, f"{side}.csv", "text/csv", key="download-csv"
        )
        return True, hamster, side, parts

    else:
        return False, False, False, parts
    
def load_abma(sideq, parts):

    #parts.append("frame")
 
    side = "LI" if sideq == "left" else "LD"
    st.markdown("## Datos ABMA")
    uploaded_file = st.file_uploader(
        "Elige el archivo CSV para comparar",
        type=["csv"],
        accept_multiple_files=False,
        help="Selecciona un archivo csv para comparar",
        key="csv_files",
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        keys = df.keys()

        rinoceronte = {}
        rinoceronte["Frame"] = df["frame"]
        for key in keys:
            if f"{side}_FILT" in key:
                if key.split()[0] in parts:
                    rinoceronte[key.split()[0]] = df[key]
       
        dfa = pd.DataFrame(rinoceronte)
        dfa = dfa.dropna()
        dfa = dfa.set_index("Frame")

        for key in dfa:
            dfa[key] = butter_lowpass_filter(data=dfa[key].tolist(), cutoff=6, fs=120, order=3)

        return True, dfa
    else:
        return False, False


def plot(dfq, dfa, parts):

    st.write(f"## QTM")
    values = st.slider(
        'Select a range of values',
        int(dfq.index[0]), int(dfq.index[-1]), (int(dfq.index[0]), int(dfq.index[-1])))
    dfqn = {}
    dfqn["Frame"] = dfq.index[values[0]-int(dfq.index[0]):values[1]-int(dfq.index[0]-1)]
    for part in parts:
        dfqn[part] = dfq[part].loc[values[0]:values[1]]
   
    dfqn = pd.DataFrame(dfqn)
    dfqn = dfqn.set_index("Frame")
    st.line_chart(dfqn)

    st.write(f"## ABMA")
    
    dfan = {}
    values = st.slider(
        'Select a range of values',
        int(dfa.index[0]), int(dfa.index[-1]), (int(dfa.index[0]), int(dfa.index[-1])))
    dfan["Frame"] = dfa.index[values[0]-int(dfa.index[0]):values[1]-int(dfa.index[0]-1)]
    for part in parts:
        dfan[part] = dfa[part].loc[values[0]:values[1]]

    for part in parts:
        if st.checkbox(f'¿Invertir {part}?'):
            dfan[part] = -dfan[part]
    dfan = pd.DataFrame(dfan)
    dfan = dfan.set_index("Frame")
    st.line_chart(dfan)





    return dfan, dfqn


def butter_lowpass_filter(data, cutoff, fs, order):
    if order == 0:
        return data
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalise frequency
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)  # Filter data
    return y


def compare(dfq, dfa, parts):
    inverse = False
    samp = st.selectbox("Selecciona la frecuencia de muestreo de QTM", (100, 120))

    for part in parts:
        st.write(f"### {part}")
        abma = dfa[part].to_list()
        qtm = dfq[part].to_list()
        if samp != 120:
            qtm = np.interp(
                np.arange(0, len(qtm), samp / 120), np.arange(0, len(qtm)), qtm
            )
        number = st.number_input(f'Inserte el desfase de {part}', min_value=-90, max_value=90, value=0, step=1, key=part)
        abmac = [x + number for x in abma]

        shft = np.argmax(signal.correlate(qtm, abmac)) - len(abmac)
        if shft < 0:
            inverse = True
            shft = np.argmax(signal.correlate(abmac, qtm)) - len(qtm)

        if inverse:
            if shft + len(dfa[part]) < len(qtm):
                fix = len(qtm)-len(dfa[part])-shft
                qtmc = qtm[shft+fix : len(qtm)]
                abmac = abmac[0 : len(qtmc)]

            else:
                abmac = abmac[shft : shft + len(dfa[part])]
                qtmc = qtm[0: len(abmac)]

        if not inverse:
            if shft + len(dfa[part]) < len(qtm):
                qtmc = qtm[shft : shft + len(dfa[part])]
            else:
                qtmc = qtm[shft : len(qtm)]
                abmac = abmac[0 : len(qtmc)]

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
        c = np.correlate(a, b, 'same')
        distance = dtw.distance(qtmc, abmac)

        dft = {
            f"QTM - {part}": qtmc,
            f"ABMA - {part}": abmac,
            # f"ERROR ABS - {part}": errabs,
            # f"RMSE - {part}": rmse,
            # f"ABMA_RAW - {part}": abma_raw,
        }


        a = ma.masked_invalid(qtmc)
        b = ma.masked_invalid(abmac)
        msk = (~a.mask & ~b.mask)
        pcoef = scipy.stats.pearsonr(a[msk], b[msk])
        scoef = scipy.stats.spearmanr(a[msk], b[msk])
        kcoef = scipy.stats.kendalltau(a[msk], b[msk])
        cohen = cohend(qtmc, abmac)
        errores ={
            "Max ABSE": max(errabs),
            "Min ABSE": min(errabs),
            "Mean ABSE": errabs.mean(),
            "Mean RMSE": rmse,
            "Max X-Corr": max(c),
            "Mean DTW": distance/len(abmac),
            "Pearson's correlation": pcoef[0],
            "Pearson's p-value": pcoef[1],
            "Spearman's correlation": scoef[0],
            "Spearman's p-value": scoef[1],
            "Kendall's correlation": kcoef[0],
            "Kendall's tau": kcoef[1],
            "QTM Shapiro's correlation": shapiro(qtmc).statistic,
            "QTM Shapiro's p-value": shapiro(qtmc).pvalue,
            "ABMA Shapiro's correlation": shapiro(abmac).statistic,
            "ABMA Shapiro's p-value": shapiro(abmac).pvalue,
            "Cohen's d": cohen,
        }

        dft = pd.DataFrame(dft)
        st.markdown(f"##### Gráficos de las señales")
        st.line_chart(dft)

        st.markdown(f"##### Correlación cruzada")
        st.line_chart(c)

        st.markdown(f"##### DTW warping path")
        path = dtw.warping_path(qtmc, abmac)
        figure, axes = dtwvis.plot_warping(qtmc, abmac, path)
        #dtwvis.plot_warping(qtmc, abmac, path, filename="warp.png")
        st.pyplot(figure)
        #st.image("warp.png", use_column_width=True)

        st.markdown(f"##### Tabla de resultados")
        st.write(errores)


        #YA,YB = np.array([qtmc, randomize(qtmc)]), np.array([abmac, randomize(np.array(abmac))])
        YA,YB = np.array(randomizeM(qtmc, 100)), np.array(randomizeM(np.array(abmac), 100))
        spm = spm1d.stats.ttest_paired(YA, YB)
        spmi = spm.inference(0.05, two_tailed=False, interp=True)
        st.write(spmi)

        #(2) Plot:
        #plt.close('all')
        ### plot mean and SD:
        fig,AX = plt.subplots( 1, 2, figsize=(8, 3.5) )
        ax     = AX[0]
        plt.sca(ax)
        spm1d.plot.plot_mean_sd(YA)
        spm1d.plot.plot_mean_sd(YB, linecolor='r', facecolor='r')
        ax.axhline(y=0, color='k', linestyle=':')
        ax.set_xlabel('Time (%)')
        ax.set_ylabel(f'{part} angle  (deg)')
        ### plot SPM results:
        ax     = AX[1]
        plt.sca(ax)
        spmi.plot()
        spmi.plot_threshold_label(fontsize=8)
        spmi.plot_p_values(size=10, offsets=[(0,0.3)])
        ax.set_xlabel('Time (%)')
        plt.tight_layout()
        st.pyplot(fig)




def usach_plot():
    merged, dfq, sideq, parts = merge_qtm()
    if merged:
        loaded, dfa = load_abma(sideq, parts)
        if loaded:
            dfan, dfqn = plot(dfq, dfa, parts)
            if st.checkbox(f'¿Comparar datos?'):
                compare(dfqn, dfan, parts)
