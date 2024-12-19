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
from PIL import Image
im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)



translate = {
    "knee": "knee",
    "hip": "hip",
    "ankle": "foot",
    "trunk": "torso"
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
    st.markdown("# Comparación QTM/ABMA-LITE 🏃‍♂️")
    st.sidebar.markdown("# Comparación QTM/ABMA-LITE 🏃‍♂️️")
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
        #plot_data = st.checkbox(f'¿Graficar datos?')
        plot_data = False
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

    side = "L" if sideq == "left" else "R"
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
            if "torso" in parts:
                rinoceronte["torso"] = df["torso_FILT"]
            if f"{side}_FILT" in key:
                if key.split("_")[0][:-1] in parts:
                    rinoceronte[key.split("_")[0][:-1]] = df[key]


        dfa = pd.DataFrame(rinoceronte)
        dfa = dfa.dropna()
        dfa = dfa.set_index("Frame")

        for key in dfa:
            if sum(dfa[key])/len(dfa[key]) >= 300:
                dfa[key] = dfa[key]-360

            if sum(dfa[key])/len(dfa[key]) <= -200:
                dfa[key] = dfa[key]+360

        return True, dfa
    else:
        return False, False


def plot(dfq, dfa, parts):
    st.write(f"## QTM")
    #values = st.slider(
    #    'Select a range of values',
    #    int(dfq.index[0]), int(dfq.index[-1]), (int(dfq.index[0]), int(dfq.index[-1])))
    values = [int(dfq.index[0]), int(dfq.index[-1])]
    dfqn = {}
    dfqn["Frame"] = dfq.index[values[0]-int(dfq.index[0]):values[1]-int(dfq.index[0]-1)]
    for part in parts:
        dfqn[part] = dfq[part].loc[values[0]:values[1]]

    dfqn = pd.DataFrame(dfqn)
    dfqn = dfqn.set_index("Frame")
    st.line_chart(dfqn)

    st.write(f"## ABMA")

    dfan = {}
    #values = st.slider(
    #    'Select a range of values',
    #    int(dfa.index[0]), int(dfa.index[-1]), (int(dfa.index[0]), int(dfa.index[-1])))
    values = [int(dfa.index[0]), int(dfa.index[-1])]
    dfan["Frame"] = dfa.index[values[0]-int(dfa.index[0]):values[1]-int(dfa.index[0]-1)]
    for part in parts:
        dfan[part] = dfa[part].loc[values[0]:values[1]]

    #for part in parts:
    #    if st.checkbox(f'¿Invertir {part}?'):
    #        dfan[part] = -dfan[part]
    #    number = st.number_input(
    #        "Ingresar desfase", value=0, placeholder="Type a number...", min_value=-360, max_value=360, step=180, key=f"{part}-desfase"
    #    )
    #    dfan[part] = dfan[part]+number


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
    #samp = st.selectbox("Selecciona la frecuencia de muestreo de QTM", (100, 120))
    samp = st.number_input(
        "Ingresar la frecuencia de muestreo de QTM", value=88, placeholder="Type a number...", min_value=0, max_value=240, step=1, key=f"freq-qtm"
    )
    adv = st.toggle(f"Modo avanzado", key=f"advanced")

    for part in parts:
        st.write(f"### {part}")
        abma = dfa[part]#.to_list()
        qtm = dfq[part]#.to_list()
        #st.write((qtm))


        if samp != 120:
            #st.warning("Oversampling")
            qtmSamp = np.interp(np.arange(0, len(qtm), samp / 120), np.arange(0, len(qtm)), qtm).tolist()
            index = []
            for i in range(len(qtmSamp)):
                index.append(i+1)

            qtmNew = pd.DataFrame({f"Frame": index, f"{part}": qtmSamp})
            qtmNew.set_index("Frame", inplace=True)
            #st.write(qtmNew)
            qtm = qtmNew[part]


        if st.checkbox(f'¿Invertir {part} QTM?'):
            qtm = -qtm
        number = st.number_input(
            "Ingresar desfase QTM", value=0, placeholder="Type a number...", min_value=-360, max_value=360, step=90, key=f"{part}-desfase-qtm"
        )
        qtm = qtm + number

        values = st.slider(
            'Seleccionar rango QTM',
            int(qtm.index[0]), int(qtm.index[-1]), (int(qtm.index[0]), int(qtm.index[-1])),
            key=f"{part}-rango-qtm",
        )

        qtm = qtm.loc[values[0]:values[1]]#.to_list()

        dfo = pd.DataFrame({f"QTM - {part}": qtm})
        st.markdown(f"###### Señal original QTM")
        st.line_chart(dfo)

        if st.checkbox(f'¿Invertir {part} ABMA?'):
            abma = -abma
        number = st.number_input(
            "Ingresar desfase ABMA", value=0, placeholder="Type a number...", min_value=-360, max_value=360, step=90, key=f"{part}-desfase-abma"
        )
        abma = abma + number

        values = st.slider(
            'Seleccionar rango ABMA',
            int(abma.index[0]), int(abma.index[-1]), (int(abma.index[0]), int(abma.index[-1])),
            key=f"{part}-rango-abma",
        )

        abma = abma.loc[values[0]:values[1]]#.to_list()

        dfo = pd.DataFrame({f"QTM - {part}": abma})
        st.markdown(f"###### Señal original ABMA")
        st.line_chart(dfo)

        try:
            abma = abma.tolist()
            qtm = qtm.tolist()

            # Compute cross-correlation
            correlation = signal.correlate(qtm, abma)

            # Find the index of the maximum correlation
            lag_index = np.argmax(correlation)

            # Calculate the actual lag
            lag = lag_index - (len(abma) - 1)

            # Now shift and sync the signals based on the lag
            if lag > 0:
                # Shift 'abma' forward (i.e., pad at the beginning)
                abma_aligned = np.pad(abma, (lag, 0), mode='constant')[:len(qtm)]
                qtm_aligned = qtm
            elif lag < 0:
                # Shift 'abma' backward (i.e., trim 'qtm' and pad 'abba')
                qtm_aligned = np.pad(qtm, (-lag, 0), mode='constant')[:len(abma)]
                abma_aligned = abma[:len(qtm)]
            else:
                # No shift needed, the signals are already aligned
                abma_aligned = abma
                qtm_aligned = qtm[:len(abma)]

            # Ensure both signals have the same length by trimming
            min_len = min(len(qtm_aligned), len(abma_aligned))
            qtm_aligned = qtm_aligned[lag:min_len]
            abma_aligned = abma_aligned[lag:min_len]


            qtmc = qtm_aligned
            abmac = abma_aligned

            dft = {
                f"QTM - {part}": qtmc,
                f"ABMA - {part}": abmac,
                # f"ERROR ABS - {part}": errabs,
                # f"RMSE - {part}": rmse,
                # f"ABMA_RAW - {part}": abma_raw,
            }

            dft = pd.DataFrame(dft)

            st.markdown(f"###### Gráficos de las señales sincronizadas")
            st.line_chart(dft)


            errabs = np.absolute(np.subtract(qtmc,abmac))
            MSE = np.square(np.subtract(qtmc,abmac)).mean()
            rmse = math.sqrt(MSE)




            a = qtmc
            b = abmac
            a = (a - np.mean(a)) / (np.std(a) * len(a))
            b = (b - np.mean(b)) / (np.std(b))
            c = np.correlate(a, b, 'same')


            if adv:
                distance = dtw.distance(qtmc, abmac)
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
            else:
                errores ={
                    "Max ABSE": max(errabs),
                    "Min ABSE": min(errabs),
                    "Mean ABSE": errabs.mean(),
                    "Mean RMSE": rmse,
                }




            st.markdown(f"###### Correlación cruzada")
            st.line_chart(c)

            if adv:

                st.markdown(f"##### DTW warping path")
                path = dtw.warping_path(qtmc, abmac)
                figure, axes = dtwvis.plot_warping(qtmc, abmac, path)
                #dtwvis.plot_warping(qtmc, abmac, path, filename="warp.png")
                st.pyplot(figure)
                #st.image("warp.png", use_column_width=True)

            st.markdown(f"##### Tabla de resultados")
            st.write(errores)

            if adv:
                st.markdown(f"##### SPM")
                #YA,YB = np.array([qtmc, randomize(qtmc)]), np.array([abmac, randomize(np.array(abmac))])
                YA,YB = np.array(randomizeM(np.array(qtmc), 100)), np.array(randomizeM(np.array(abmac), 100))
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

        except:
            st.warning("Error al sincronizar las señales")




def usach_plot2():
    merged, dfq, sideq, parts = merge_qtm()
    if merged:
        loaded, dfa = load_abma(sideq, parts)
        if loaded:
            dfan, dfqn = plot(dfq, dfa, parts)

            compare(dfqn, dfan, parts)


def sync_signals(qtm, abma):
    # Get the lengths of the signals
    len_qtm = len(qtm)
    len_abma = len(abma)

    # Determine the length of the synchronized signals
    if len_qtm > len_abma:
        # Trim qtm
        qtm_trimmed = qtm[:len_abma]
        abma_padded = np.pad(abma, (0, len_qtm - len_abma), mode='constant')
        return qtm_trimmed, abma_padded
    elif len_abma > len_qtm:
        # Trim abma
        abma_trimmed = abma[:len_qtm]
        qtm_padded = np.pad(qtm, (0, len_abma - len_qtm), mode='constant')
        return qtm_padded, abma_trimmed
    else:
        # Both signals are already the same length
        return qtm, abma


def main():
    usach_plot2()

if __name__ == "__main__":
    main()
