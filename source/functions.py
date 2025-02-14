# -*- coding: utf-8 -*-
"""
@file    : functions.py
@brief   : Handles program functions
@date    : 2024/08/22
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import asyncio
import os
import subprocess
import time
from datetime import datetime
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pytz
import streamlit as st
from pandas.api.types import (is_categorical_dtype, is_datetime64_any_dtype,
                              is_numeric_dtype, is_object_dtype)
from PIL import Image
from plotly.subplots import make_subplots



pio.templates.default = 'plotly'
pio.templates[pio.templates.default].layout.colorway = px.colors.qualitative.Plotly

trim = 3


def search_db_rut(rutdv):
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath, index_col='ID')
    df2 = pd.read_csv(dataPath)
    data = df.loc[df['RUT'] == rutdv]
    data2 = df2.loc[df2['RUT'] == rutdv]
    return data, data2


def search_db_rut_last(rutdv):
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath)  # pd.read_csv(dataPath, index_col="ID")
    data = df.loc[df['RUT'] == rutdv]
    return data


def search_db_fecha_last(fecha):
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(
        dataPath
    )  # pd.read_csv(dataPath, index_col="ID")pd.read_csv(dataPath, index_col="ID")
    data = df.loc[df['FECHA'] == str(fecha)]
    return data


def list_db_rut():
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath, index_col='ID')
    data = sorted(df['RUT'].tolist())
    return list(dict.fromkeys(data))


def plot_exam(examid):
    df = open_exam(examid)
    # st.line_chart(data=df, x="timestamps", use_container_width=True)

    x = df['timestamps'].tolist()
    t = []
    for i in x:
        t.append(datetime.fromtimestamp(i))

    fig = make_subplots(rows=1, cols=1)
    for i in df:
        if i != 'timestamps' and i != 'Right AUX':
            fig.add_trace(go.Scatter(x=t, y=df[i], name=f'{i}', line={'width': 2}),
                          row=1, col=1)

    fig.update_layout(
        font=dict(size=14),
        title=dict(
            text=f'Resultados EEG<br><sup><i>{examid}</i></sup>',
            font=dict(size=25),
            yref='paper',
        ),
        xaxis_title_text='<b>Hora</b> [s]',
        yaxis_title_text='<b>Tensión</b> [μV]',
    )

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def plot_df(df):
    x = df['timestamps'].tolist()
    t = []
    for i in x:
        t.append(datetime.fromtimestamp(i))
    fig = make_subplots(rows=1, cols=1)
    for i in df:
        if i != 'timestamps' and i != 'Right AUX':
            fig.add_trace(go.Scatter(x=t, y=df[i], name=f'{i}', line={'width': 2}),
                          row=1, col=1)

    fig.update_layout(
        font=dict(size=14),
        title=dict(
            text='Resultados EEG<br><sup><i>Datos capturados</i></sup>',
            font=dict(size=25),
            yref='paper',
        ),
        xaxis_title_text='<b>Tiempo</b> [s]',
        yaxis_title_text='<b>Tensión</b> [mV]',
    )

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def list_db_id():
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath)
    data = df['ID'].tolist()
    return data


def open_exam(examid):
    dataPath = f'assets/csv/{examid}.csv'
    df = pd.read_csv(dataPath)
    return df


def plot_exam_full(examid):
    df = open_exam(examid)
    # st.line_chart(data=df, x="timestamps", use_container_width=True)

    x = df['timestamps'].tolist()
    t = []
    for i in x:
        t.append(datetime.fromtimestamp(i))

    size = 1000
    fig = make_subplots(rows=4, cols=1)
    row = 1
    for i in df:
        if i != 'timestamps' and i != 'Right AUX':
            fig.add_trace(go.Scatter(x=t, y=df[i], name=f'{i}', line={'width': 2}),
                          row=row, col=1)
            fig.update_xaxes(title_text='<b>Hora</b> [-]', row=row, col=1)
            fig.update_yaxes(title_text='<b>Tensión</b> [μV]', row=row, col=1)
            row += 1

    fig.update_layout(
        font=dict(size=14),
        title=dict(
            text=f'Resultados EEG<br><sup><i>{examid}</i></sup>',
            font=dict(size=25),
            yref='paper',
        ),
        height=size,
        # xaxis_title_text=f"<b>Hora</b> [s]",
        # yaxis_title_text=f"<b>Tensión</b> [μV]",
    )

    st.plotly_chart(fig, height=size, theme='streamlit', use_container_width=True)


def unique(list1):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list


def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    return (-s) % 11


def search_db():
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath, index_col='ID')
    df2 = pd.read_csv(dataPath)
    return df, df2


def search_db_fecha(fecha):
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath, index_col='ID')
    df2 = pd.read_csv(dataPath)
    data = df.loc[df['FECHA'] == str(fecha)]
    data2 = df2.loc[df2['FECHA'] == str(fecha)]
    return data, data2


def display_data(data):
    st.dataframe(data, use_container_width=True)


def validate_data(data):
    if data.shape[1] > 0:
        if data.shape[0] > 0:
            return True
        else:
            st.warning('##### No se encontraron exámenes para ese RUT')
            return False


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns
    -------
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox('Add filters')

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect('Filter dataframe on', df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f'Values for {column}',
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f'Values for {column}',
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(f'Substring or regex in {column}', )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def title(fig, text):
    fig.update_layout(
        font=dict(size=14),
        title=dict(text=f'{text}', font=dict(size=25), yref='paper'),
        # xaxis_title_text=x,
        # yaxis_title_text=y,
    )
    return fig


def validate_date(fecha):
    format = '%d-%m-%Y'

    # checking if format matches the date
    res = True

    # using try-except to check for truth value
    try:
        res = bool(datetime.strptime(fecha, format))
    except ValueError:
        st.warning('Escriba la fecha en el formato requerido')
        res = False

    return res


def validate_rut(rutdv):
    try:
        rut = int(rutdv.split('-')[0].replace('.', ''))
        dv = int(rutdv.split('-')[1])

        if dv == digito_verificador(rut):
            # st.success('El RUT es válido')

            if st.button(
                    'Buscar RUT',
                    type='primary',
                    # help="Buscar por RUT",
            ):
                data, data2 = search_db_rut(rutdv)
                st.session_state['data'], st.session_state['data2'] = data, data2
            # display_data(data)

        else:
            st.warning('##### El RUT es inválido')
    except Exception:
        st.error('##### Escriba el RUT en el formato requerido')


def validate_fecha(fecha, manual2):
    if manual2:
        if validate_date(fecha):
            if st.button(
                    'Buscar Fecha',
                    type='primary',
                    # help="Buscar por Fecha",
            ):
                data, data2 = search_db_fecha(fecha)
                st.session_state['data'], st.session_state['data2'] = data, data2
    if not manual2:
        if st.button(
                'Buscar Fecha',
                type='primary',
                # help="Buscar por Fecha",
        ):
            data, data2 = search_db_fecha(fecha)
            st.session_state['data'], st.session_state['data2'] = data, data2


def mod_rut(rutdvO, cont, placeholder):
    rutdvO = val_rut(rutdvO)
    cont += 1
    if rutdvO is not None:
        if rutdvO[:-1].isdigit() and len(rutdvO) > 1:
            placeholder.empty()
            rut = int(rutdvO[:-1])
            dv = int(rutdvO[-1].replace('k', '10').replace('K', '10'))
            value = f'{int(rutdvO[:-1]):_}'.replace('_', '.')
            value = f'{value}-{rutdvO[-1]}'
            rutdv = value
            placeholder = st.empty()
            label = (
                'Ingrese RUT'  # r"$\textsf{\large Ingrese RUT sin puntos ni guión}$"
            )
            rutdvN = placeholder.text_input(
                label,
                value=value,
                label_visibility='visible',
                disabled=False,
                placeholder=value,
                key=f'rut{cont}',
                help='RUT del paciente',
            )
            if rutdvN != value:
                # placeholder.empty()
                mod_rut(rutdvN, cont, placeholder)
            else:
                if dv == digito_verificador(rut):
                    st.session_state['rutdv'] = rutdv
                    data, data2 = search_db_rut(rutdv)
                    st.session_state['data'], st.session_state['data2'] = data, data2
                    if validate_data(
                            st.session_state['data']) and (rutdv):  # or fecha):
                        data = st.session_state['data']
                        # df = data.drop('ID', axis=0)
                        st.dataframe(data)
                        # display_results_general(st.session_state["data"])

                else:
                    st.warning('##### El RUT es inválido')
        else:
            st.error('##### Escriba el RUT en el formato requerido')


def mod_rut_2(rutdvO, cont, placeholder):
    cont += 1
    rutdvO = val_rut(rutdvO)
    if rutdvO is not None:
        if rutdvO[:-1].isdigit() and len(rutdvO) > 1:
            placeholder.empty()
            rut = int(rutdvO[:-1])
            dv = int(rutdvO[-1].replace('k', '10').replace('K', '10'))
            value = f'{int(rutdvO[:-1]):_}'.replace('_', '.')
            value = f'{value}-{rutdvO[-1]}'
            rutdv = value
            placeholder = st.empty()
            label = (
                'Ingrese RUT'  # r"$\textsf{\large Ingrese RUT sin puntos ni guión}$"
            )
            rutdvN = placeholder.text_input(
                label,
                value=value,
                label_visibility='visible',
                disabled=False,
                placeholder=value,
                key=f'rut{cont}',
                help='RUT del paciente',
            )
            if rutdvN != value:
                # placeholder.empty()
                mod_rut_2(rutdvN, cont, placeholder)
            else:
                if dv == digito_verificador(rut):
                    st.session_state['rutdv'] = rutdv
                    st.session_state['valid'] = True
                else:
                    st.warning('El RUT es inválido')
                    st.session_state['valid'] = False
        else:
            st.error('Escriba el RUT en el formato requerido')
            st.session_state['valid'] = False


def clear_rut(rutdvO, cont, placeholder):
    cont += 1
    rutdvO = val_rut(rutdvO)
    if rutdvO is not None:
        if rutdvO[:-1].isdigit() and len(rutdvO) > 1:
            placeholder.empty()
            rut = int(rutdvO[:-1])
            dv = int(rutdvO[-1].replace('k', '10').replace('K', '10'))
            value = f'{int(rutdvO[:-1]):_}'.replace('_', '.')
            value = f'{value}-{rutdvO[-1]}'
            rutdv = value
            placeholder = st.empty()
            label = 'Ingrese RUT sin puntos ni guión'  # r"$\textsf{\large Ingrese RUT sin puntos ni guión}$"
            rutdvN = placeholder.text_input(
                label,
                value=value,
                label_visibility='visible',
                disabled=False,
                placeholder=value,
                key=f'rutclear{cont}',
                help='RUT del paciente',
            )
            if rutdvN != value:
                # placeholder.empty()
                mod_rut_2(rutdvN, cont, placeholder)
            else:
                if dv == digito_verificador(rut):
                    st.session_state['rutdv'] = rutdv
                    st.session_state['valid'] = True
                else:
                    st.warning('El RUT es inválido')
                    st.session_state['valid'] = False
        else:
            st.error('Escriba el RUT en el formato requerido')
            st.session_state['valid'] = False


def val_rut(rutdv):
    # rutdv = rutdv.replace("k", "0").replace("K", "0")
    x = [i for i in rutdv]
    vali = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '-', 'k', 'K']
    res = [ele for ele in x if (ele not in vali)]

    if len(res) > 0:
        banana = ''
        for y in sorted(set(res)):
            banana += ' ' + y + ','
        banana = banana[:-1]
        st.error(f'##### Character no admitido: {banana}')
    else:
        rutdv = rutdv.replace('-', '').replace('.', '')
        # rut = int(rutdv[:-1])
        # dv = int(rutdv[-1])
        # if dv == digito_verificador(rut):
        return rutdv


def avm_start():
    time.sleep(1)
    return True


def avm_stop():
    time.sleep(1)
    return True


def trigger():
    if not st.session_state['no_sound']:
        tone = generate_tone(
            st.session_state['frequency'],
            st.session_state['duration'],
            st.session_state['sample_rate'],
        )
        play_tone(tone, st.session_state['sample_rate'])

    ts = datetime.now(pytz.timezone('America/Santiago')).timestamp()

    if not st.session_state['no_trigger']:
        relay = pyhid_usb_relay.find()
        if st.session_state['invert_trigger'] and not st.session_state['banana']:
            relay.toggle_state(1)
            relay.toggle_state(2)
            time.sleep(1)
            st.session_state['banana'] = True
        
        if not st.session_state['invert_trigger'] and st.session_state['banana']:
            relay.toggle_state(1)
            relay.toggle_state(2)
            time.sleep(1)
            st.session_state['banana'] = False
        
        relay.toggle_state(1)
        relay.toggle_state(2)
        time.sleep(st.session_state['trigg'] / 1000)
        relay.toggle_state(1)
        relay.toggle_state(2)

    return ts


def generate_tone(frequency, duration, sample_rate=44100):
    # Generate time values
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate the tone (a sine wave of the given frequency)
    tone = np.sin(2 * np.pi * frequency * t)

    return tone


def play_tone(tone, sample_rate=44100):
    sd.play(tone, samplerate=sample_rate)
    sd.wait()  # Wait until the sound has finished playing


def start_stop_rec(recording, noAvm):
    # Pin Definitions
    try:
        if not recording:
            with st.spinner('Iniciando grabación'):
                dtnow = datetime.now(pytz.timezone('America/Santiago'))
                # st.session_state["tsi"] = dtnow.timestamp()
                st.session_state['fechai'] = dtnow.strftime('%Y-%m-%d')
                st.session_state['horai'] = dtnow.strftime('%H:%M:%S')
                ts = trigger()
                st.session_state['tsi'] = ts
                st.success('##### Grabación iniciada')
            st.session_state['recording'] = True
            return True

        if recording:
            with st.spinner('Deteniendo grabación'):
                dtnow = datetime.now(pytz.timezone('America/Santiago'))
                # st.session_state["tsf"] = dtnow.timestamp()
                st.session_state['fechaf'] = dtnow.strftime('%Y-%m-%d')
                st.session_state['horaf'] = dtnow.strftime('%H:%M:%S')
                ts = trigger()
                st.session_state['tsf'] = ts
                st.error('##### Grabación detenida')
            st.session_state['recording'] = False
            st.session_state['examid'] = (
                f'{st.session_state["rutdv"]} - {st.session_state["fechai"]} - {st.session_state["horai"]}'
            )
            return True
    except Exception:
        st.warning('##### No se encuentra trigger, reconecte USB')
        return False


def update_db(dataDict):
    dataPath = 'assets/db/db.csv'
    df = pd.read_csv(dataPath)
    df = pd.concat([df, dataDict])
    df.to_csv(f'{dataPath}', index=False)


def save_df():
    dataDict = {
        # "ID": st.session_state["examid"],
        'RUT': st.session_state['rutdv'],
        'FECHA': st.session_state['fechai'],
        'HORA I': st.session_state['horai'],
        'TIMESTAMP I': st.session_state['tsi'],
        'HORA F': st.session_state['horaf'],
        'TIMESTAMP F': st.session_state['tsf'],
    }
    dfOut = pd.DataFrame.from_records([dataDict])
    update_db(dfOut)


def find_avm():
    avms = []
    try:

        with st.spinner('Buscando dispositivos...'):
            time.sleep(1)
            avms = [
                {
                    'name': 'AVM-1',
                    'address': '00:01:02:03:01',
                },
                {
                    'name': 'AVM-2',
                    'address': '00:01:02:03:02',
                },
                {
                    'name': 'AVM-3',
                    'address': '00:01:02:03:03',
                },
                {
                    'name': 'AVM-4',
                    'address': '00:01:02:03:04',
                },
                {
                    'name': 'AVM-5',
                    'address': '00:01:02:03:05',
                },
            ]  # list_muses()

        if len(avms) == 0:
            st.warning(
                '##### No se encontraron dispositivos, apague y encienda el AVM y realice una nueva búsqueda'
            )
    except Exception:
        st.warning(
            '##### Bluetooth apagado, encienda el bluetooth del computador y realice una nueva búsqueda'
        )

    return avms


def find_muse():
    muses = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        with st.spinner('Buscando dispositivos...'):
            # muses = list_muses()
            if len(muses) == 0:
                st.warning(
                    '##### No se encontraron dispositivos, apague y encienda el Muse y realice una nueva búsqueda'
                )
    except Exception:
        st.warning(
            '##### Bluetooth apagado, encienda el bluetooth del computador y realice una nueva búsqueda'
        )

    return muses


def set_session():
    fecha = datetime.now(pytz.timezone('America/Santiago')).strftime('%Y-%m-%d')
    hora = datetime.now(pytz.timezone('America/Santiago')).strftime('%H:%M:%S')
    timestamp = datetime.now(pytz.timezone('America/Santiago')).timestamp()
    if 'new' not in st.session_state:
        st.session_state['new'] = False
    if 'examid' not in st.session_state:
        st.session_state['examid'] = (
            ''  # list_db_id()[-1] if len(list_db_id()) > 0 else None
        )
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False
    if 'started' not in st.session_state:
        st.session_state['started'] = False
    if 'rutdv' not in st.session_state:
        st.session_state['rutdv'] = ''
    if st.session_state['rutdv'] != '':
        st.session_state['rutdv'] = ''
    if 'valid' not in st.session_state:
        st.session_state['valid'] = False
    if 'recorded' not in st.session_state:
        st.session_state['recorded'] = False
    if 'avms' not in st.session_state:
        st.session_state['avms'] = []
    if 'muses' not in st.session_state:
        st.session_state['muses'] = []
    if 'fecha' not in st.session_state:
        st.session_state['fecha'] = fecha
    if 'fechai' not in st.session_state:
        st.session_state['fechai'] = fecha
    if 'horai' not in st.session_state:
        st.session_state['horai'] = hora
    if 'tsi' not in st.session_state:
        st.session_state['tsi'] = timestamp
    if 'avmi' not in st.session_state:
        st.session_state['avmi'] = timestamp
    if 'delsysi' not in st.session_state:
        st.session_state['delsysi'] = timestamp
    if 'eegi' not in st.session_state:
        st.session_state['eegi'] = timestamp
    if 'fechaf' not in st.session_state:
        st.session_state['fechaf'] = fecha
    if 'horaf' not in st.session_state:
        st.session_state['horaf'] = hora
    if 'tsf' not in st.session_state:
        st.session_state['tsf'] = timestamp
    if 'avmf' not in st.session_state:
        st.session_state['avmf'] = timestamp
    if 'delsysf' not in st.session_state:
        st.session_state['delsysf'] = timestamp
    if 'eegf' not in st.session_state:
        st.session_state['eegf'] = timestamp
    if 'cont' not in st.session_state:
        st.session_state['cont'] = 0
    if 'trigg' not in st.session_state:
        st.session_state['trigg'] = 100
    if 'device' not in st.session_state:
        st.session_state['device'] = '-'
    if 'deviceM' not in st.session_state:
        st.session_state['deviceM'] = '-'
    if 'noAvm' not in st.session_state:
        st.session_state['noAvm'] = True
    if 'noMuse' not in st.session_state:
        st.session_state['noMuse'] = False
    if 'streaming' not in st.session_state:
        st.session_state['streaming'] = False
    if 'duration' not in st.session_state:
        st.session_state['duration'] = config.Config.Duration.default
    if 'frequency' not in st.session_state:
        st.session_state['frequency'] = config.Config.Tone.default
    if 'sample_rate' not in st.session_state:
        st.session_state['sample_rate'] = config.Config.Tone.sample_rate
    if 'no_sound' not in st.session_state:
        st.session_state['no_sound'] = config.Config.Default.no_sound
    if 'no_sound' in st.session_state:
        st.session_state['no_sound'] = config.Config.Default.no_sound
    if 'no_trigger' not in st.session_state:
        st.session_state['no_trigger'] = config.Config.Default.no_trigger
    if 'no_trigger' in st.session_state:
        st.session_state['no_trigger'] = config.Config.Default.no_trigger
    if 'avanzado' not in st.session_state:
        st.session_state['avanzado'] = config.Config.Default.avanzado


def reset_session():
    st.session_state['avanzado'] = config.Config.Default.avanzado
    st.session_state['trigg'] = config.Config.Trigger.default
    st.session_state['duration'] = config.Config.Duration.default
    st.session_state['frequency'] = config.Config.Tone.default
    st.session_state['no_sound'] = config.Config.Default.no_sound
    st.session_state['no_trigger'] = config.Config.Default.no_trigger


def input_rut(placeholder):
    placeholder.text_input(
        label='Ingrese RUT',  # r"$\textsf{\large Ingrese RUT sin puntos ni guión}$",
        value=st.session_state.rutdv,
        label_visibility='visible',
        disabled=False,
        # placeholder="123456789",
        key=f"ruts{st.session_state['cont']}",
        help='RUT del paciente',
    )


def rut_input(show_buttons=True):

    try:
        st.session_state.rutdv
    except Exception:
        set_session()

    placeholder = st.empty()
    rutdv = placeholder.text_input(
        label=r'$\textsf{\large Ingrese RUT}$',
        value=st.session_state.rutdv,
        label_visibility='visible',
        disabled=False,
        # placeholder="123456789",
        key=f"ruts{st.session_state['cont']}",
        help='RUT del paciente',
    )

    if rutdv:
        mod_rut_2(rutdv, st.session_state['cont'], placeholder)
        rutdv = st.session_state['rutdv']
        if show_buttons:
            st.session_state['recorded'] = False
            if st.session_state['valid']:
                if st.session_state['rutdv'] != '':
                    # st.session_state["noAvm"] = st.toggle("¿Continuar sin un AVM?")
                    st.session_state['noMuse'] = (
                        True  # st.toggle("¿Continuar sin un Muse?")
                    )
                    if st.session_state['noAvm'] is True:
                        st.session_state['device'] = '-'
                    if st.session_state['noMuse'] is True:
                        st.session_state['deviceM'] = '-'
                    # if st.session_state["noAvm"] == False or st.session_state["noMuse"] == False:
                    #    if st.button("Buscar Dispositivos", type="primary"):
                    #        if st.session_state["noAvm"] == False:
                    #            st.session_state["avms"] = find_avm()
                    #        if st.session_state["noMuse"] == False:
                    #            st.session_state["muses"] = find_avm()


def set_form():
    if (len(st.session_state['avms']) > 0
            and st.session_state['rutdv']) or (st.session_state['noAvm'] is True
                                               and st.session_state['rutdv']):
        st.session_state['avanzado'] = st.toggle('Opciones avanzadas', False)

        if st.session_state['avanzado']:
            st.session_state['trigg'] = st.number_input(
                'Duración Trigger (ms)',
                min_value=config.Config.Trigger.min,
                max_value=config.Config.Trigger.max,
                value=st.session_state['trigg'],
                step=config.Config.Trigger.step,
                help='Duración del trigger en milisegundos',
            )
            st.session_state['duration'] = st.number_input(
                'Duración Tono [s]',
                min_value=config.Config.Duration.min,
                max_value=config.Config.Duration.max,
                value=st.session_state['duration'],
                step=config.Config.Duration.step,
                help='Duración del tono en segundos',
            )
            st.session_state['frequency'] = st.number_input(
                'Frecuencia Tono (Hz)',
                min_value=config.Config.Tone.min,
                max_value=config.Config.Tone.max,
                value=st.session_state['frequency'],
                step=config.Config.Tone.step,
                help='Frecuencia del tono en Hertz',
            )
            st.session_state['no_sound'] = st.toggle('No Sound', False)
            st.session_state['no_trigger'] = st.toggle('No Trigger', False)
            st.session_state['invert_trigger'] = st.toggle('Invert Trigger', False)
            if 'banana' not in st.session_state:
                st.session_state['banana'] = False


        if not st.session_state['avanzado']:
            reset_session()

        submit = st.button(
            'Iniciar/Detener examen', type='primary'
        )  # st.form_submit_button(f"Iniciar/Detener examen", type="primary")
        if submit:
            success = start_stop_rec(st.session_state['recording'],
                                     st.session_state['noAvm'])
            if success:
                if st.session_state['tsf'] > st.session_state['tsi']:
                    with st.spinner('Guardando información...'):
                        time.sleep(1)
                        save_df()


def show_data(mode):
    datos = []
    if mode == 'rut':
        datos = search_db_rut_last(st.session_state['rutdv'])
    if mode == 'fecha':
        datos = search_db_fecha_last(st.session_state['fecha'])

    if datos.shape[0] > 0:
        datosInv = datos.iloc[::-1]
        st.dataframe(datosInv, use_container_width=True)


def clear_page(title='Lanek'):
    try:
        im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            page_icon=im,
            layout='wide',
        )

        # add_logo("assets/logos/ap75.png", height=75)

        hide_streamlit_style = """
            <style>
                .reportview-container {
                    margin-top: -2em;
                }
                #MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                #stDecoration {display:none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # with open("./source/style.css") as f:
        #    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def load_csv(key):
    # st.markdown("## Datos ABMA")
    uploaded_file = st.file_uploader(
        'Elige un archivo CSV',
        type=['csv'],
        accept_multiple_files=False,
        help='Selecciona un archivo csv para comparar',
        key=key,
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.keys()

        dfa = pd.DataFrame(df)
        dfa = dfa.dropna()
        # dfa = dfa.set_index("X[s]")

        return True, dfa
    else:
        return False, False


def normalize_signal(signal, max_value, norm_value):
    normalized_signal = (signal / max_value) * norm_value
    return normalized_signal


def compute_fft(signal, sample_rate):
    # Number of samples in the signal
    N = len(signal)

    # Compute the FFT
    fft_values = np.fft.fft(signal)

    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(N, d=1 / sample_rate)

    # Compute the magnitude of the FFT values
    magnitudes = np.abs(fft_values)

    return freqs, magnitudes


def compute_statistics(freqs, magnitudes):
    # Only consider the positive half of the frequency spectrum
    positive_freqs = freqs[freqs >= 0]
    positive_magnitudes = magnitudes[freqs >= 0]

    # Compute weighted average frequency
    avg_freq = np.average(positive_freqs, weights=positive_magnitudes)

    # Compute median frequency
    cumulative_sum = np.cumsum(positive_magnitudes)
    median_freq = positive_freqs[np.searchsorted(cumulative_sum,
                                                 cumulative_sum[-1] / 2)]

    # Compute other basic statistics
    max_freq = positive_freqs[np.argmax(positive_magnitudes)]
    min_freq = positive_freqs[np.argmin(positive_magnitudes)]

    # Compute quartiles
    q1 = np.percentile(positive_freqs, 25)
    q3 = np.percentile(positive_freqs, 75)

    return {
        'average_frequency': avg_freq,
        'median_frequency': median_freq,
        'max_frequency': max_freq,
        'min_frequency': min_freq,
        'q1_frequency': q1,
        'q3_frequency': q3,
    }


def calculate_moving_average(signal, window_size_ms=50, sampling_rate_hz=1000):
    """
    Calculate the moving average of a function with a specified window size.

    Parameters
    ----------
    data (list or np.array): The input data series.
    window_size_ms (int): The size of the moving window in milliseconds.
    sampling_rate_hz (int): The sampling rate of the data in Hz (samples per second).

    Returns
    -------
    np.array: The moving average of the input data.
    """
    # Convert window size from milliseconds to number of samples
    window_size_samples = int(window_size_ms * sampling_rate_hz / 1000)

    # Create a Pandas Series from the data
    data_series = pd.Series(signal)

    # Calculate the moving average using a rolling window
    moving_average = data_series.rolling(window=window_size_samples,
                                         min_periods=1).mean()

    return moving_average.to_numpy()


def compare_signals(signal1, signal2):
    """
    Compare two signals element-wise and return a mask where:
    - 1 indicates that the value in the first signal is greater than the value in the second signal.
    - 0 indicates otherwise.

    Parameters
    ----------
    signal1 (list or np.array): The first signal data.
    signal2 (list or np.array): The second signal data.

    Returns
    -------
    np.array: A mask array with 1s and 0s based on the comparison.
    """
    # Convert signals to NumPy arrays if they aren't already
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    # Ensure both signals have the same length
    if signal1.shape != signal2.shape:
        raise ValueError('Signals must have the same length')

    # Create the mask by comparing the two signals
    mask = (signal1 > signal2).astype(int)

    return mask


def rms_rectify_signal(signal, sample_rate=1000, window_size_ms=125, overlap_ms=25):
    # Convert window size and overlap from milliseconds to samples
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    overlap_samples = int(overlap_ms * sample_rate / 1000)
    step_size = max(window_size_samples - overlap_samples, 1)

    rms_values = []
    times = []

    for start in range(0, len(signal) - window_size_samples + 1, step_size):
        window = signal[start:start + window_size_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)

        # Calculate the time corresponding to the start of the window
        time = start / sample_rate  # Time in seconds
        times.append(time)

    return np.array(rms_values), np.array(times)


def calculate_sampling_frequency(time_vector):
    """
    Calculate the sampling frequency from a time vector.

    Parameters
    ----------
    time_vector (list or np.array): The time vector, which should be in ascending order.

    Returns
    -------
    float: The sampling frequency in Hz.
    """
    # Convert time vector to a NumPy array if it isn't already
    time_vector = np.asarray(time_vector)

    # Ensure the time vector is sorted
    if not np.all(np.diff(time_vector) >= 0):
        raise ValueError('Time vector must be in ascending order')

    # Compute time differences between consecutive samples
    time_diffs = np.diff(time_vector)

    # Compute the average time interval
    avg_time_interval = np.mean(time_diffs)

    # Calculate the sampling frequency (in Hz)
    sampling_frequency = 1 / avg_time_interval

    return sampling_frequency


def check_signal_threshold_old(signal, threshold, sample_rate, window_ms=50):
    """
    Check if a signal is greater than a threshold in 50ms windows and return the corresponding time vector.

    Parameters
    ----------
    signal (np.ndarray): The input signal.
    threshold (float): The threshold value.
    sample_rate (int): The sampling rate of the signal in Hz.
    window_ms (int): The window size in milliseconds. Default is 50ms.

    Returns
    -------
    Tuple[List[bool], np.ndarray]: A tuple containing a list of booleans indicating if the signal exceeds the threshold
                                   in each window and a time vector for the start of each window.
    """
    window_size = int(sample_rate * window_ms / 1000)  # Convert window size to samples
    num_windows = len(signal) // window_size

    exceeds_threshold = []
    time_vector = np.arange(num_windows) * (window_size / sample_rate)

    for i in range(num_windows):
        window = signal[i * window_size:(i + 1) * window_size]
        exceeds_threshold.append(np.any(window > threshold))

    return exceeds_threshold, time_vector


def check_signal_threshold(time: np.ndarray, signal: np.ndarray, threshold: float,
                           window_ms: int = 50):
    """
    Check if a signal exceeds a threshold within 50ms windows and return the corresponding time vector.

    Parameters
    ----------
    time (np.ndarray): The time vector of the signal.
    signal (np.ndarray): The magnitude of the signal.
    threshold (float): The threshold value.
    window_ms (int): The window size in milliseconds. Default is 50ms.

    Returns
    -------
    Tuple[List[bool], np.ndarray, np.ndarray]: A tuple containing a list of booleans indicating if the signal exceeds
                                               the threshold in each window, a time vector for the start of each window,
                                               and an array of 1s and 0s corresponding to the boolean array.
    """
    sample_rate = 1 / (time[1] - time[0]
                       )  # Calculate the sample rate from the time vector
    window_size = int(sample_rate * window_ms / 1000)  # Convert window size to samples
    num_windows = len(signal) // window_size

    exceeds_threshold = []
    time_vector = np.arange(num_windows) * (window_size / sample_rate)
    binary_array = np.zeros(num_windows, dtype=int)

    for i in range(num_windows):
        window = signal[i * window_size:(i + 1) * window_size]
        threshold_exceeded = np.any(window > threshold)
        exceeds_threshold.append(threshold_exceeded)
        binary_array[i] = int(threshold_exceeded)

    return exceeds_threshold, time_vector, binary_array


def remove_dc_offset(signal):
    """
    Remove the DC offset from the given signal.

    Parameters
    ----------
    signal (np.ndarray): Input signal as a NumPy array.

    Returns
    -------
    np.ndarray: Signal with the DC offset removed.
    """
    # Calculate the DC offset (mean of the signal)
    dc_offset = np.mean(signal)

    # Remove the DC offset
    signal_no_dc = signal - dc_offset

    return signal_no_dc


def plot_signal(data, x='x', y='y', xaxis_title='Time (s)', yaxis_title='Signal'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data[x], y=data[y], mode='lines', name='Sine Wave'))
    st.empty()
    # Use the Plotly `relayout` event to capture zoom/pan actions
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        # xaxis=dict(
        #    rangeslider=dict(visible=True),
        #    rangeselector=dict(
        #        buttons=list([
        #            dict(count=6, label="6m", step="month", stepmode="backward"),
        #            dict(count=1, label="1m", step="month", stepmode="backward"),
        #            dict(step="all")
        #        ])
        #    )
        # ),
        # clickmode='event+select'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


def plot_signal2(
    data,
    data2,
    threshold,
    x='x',
    y='y',
    x2='x2',
    y2='y2',
    xt='Time (s)',
    yt='Signal',
    xt2='Time (s)',
    yt2='Signal',
):
    """
    Plot the signal with two traces using Plotly and Streamlit, and adds a horizontal threshold line.

    Parameters
    ----------
    data (dict): Dictionary containing the data to plot.
    data2 (dict): Dictionary containing the data for the second trace.
    threshold (float): The threshold value for the horizontal line.
    x (str): Key for the x-axis data in the dictionary.
    y (str): Key for the first y-axis data in the dictionary.
    y2 (str): Key for the second y-axis data in the dictionary.
    xaxis_title (str): Title for the x-axis.
    yaxis_title (str): Title for the y-axis.
    """
    fig = go.Figure()

    # First trace
    fig.add_trace(go.Scatter(x=data[x], y=data[y], mode='lines', name='Señal'))

    # Second trace
    fig.add_trace(
        go.Scatter(x=data2[x2], y=data2[y2] * np.max(data[y]), mode='lines',
                   name='On/Off'))

    # Horizontal threshold line
    fig.add_hline(y=threshold, line=dict(color='red', width=2, dash='dash'),
                  name='Threshold')

    fig.update_layout(
        # title='Line Chart with Plotly',
        xaxis_title=xt,
        yaxis_title=yt,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


def rectify_signal(signal):
    return np.abs(signal)


def detect_threshold_crossings(time, signal, threshold, window_size_ms, overlap_ms):
    sample_rate = 1 / (time[1] - time[0])
    # Convert window size and overlap from milliseconds to samples
    window_size_samples = int(window_size_ms * sample_rate / 1000)
    overlap_samples = int(overlap_ms * sample_rate / 1000)
    step_size = max(window_size_samples - overlap_samples, 1)

    # Initialize the result array with zeros
    threshold_crossings = np.zeros_like(signal, dtype=int)
    crossing_times = []

    for start in range(0, len(signal) - window_size_samples + 1, step_size):
        window = signal[start:start + window_size_samples]

        # Check if the signal exceeds the threshold for the entire window
        if np.all(window > threshold):
            # Mark the corresponding section in the result array as 1
            threshold_crossings[start:start + window_size_samples] = 1

            # Calculate the midpoint of the window in the time vector
            midpoint_time = time[start + window_size_samples // 2]
            crossing_times.append(midpoint_time)

    return threshold_crossings, np.linspace(time[0], time[-1], len(threshold_crossings))


def process_max():
    st.markdown('#### Selecciona el archivo de máximos')
    data, df = load_csv('maximos')
    if data:
        showOr = st.toggle('¿Mostrar señal original?')

        keys = df.keys()
        muscles = []
        maximos = {}
        for key in keys:
            if 'EMG' in key:
                muscles.append(key)
        if showOr:
            st.dataframe(df)
            st.write(muscles)

        cont = 0
        for muscle in muscles:
            st.write(f'### {muscle}')
            signal0 = remove_dc_offset(df[muscle])
            if cont > 0:
                time0 = df[f'X[s].{cont}']
            else:
                time0 = df['X[s]']
            cont += 4
            sr = calculate_sampling_frequency(time0)

            if showOr:
                st.write('#### Señal original')
                data = pd.DataFrame({'x': time0, 'y': signal0})
                plot_signal(data)

            values = st.slider(
                'Seleccionar rango de muestras',
                0,
                len(signal0),
                (0, len(signal0)),
                key=f'SLIDER_{muscle}',
            )

            signal = signal0[values[0]:values[1]]
            time0[values[0]:values[1]]
            delay = time0[values[0]]
            signalRMS, time = rms_rectify_signal(signal=signal, sample_rate=sr,
                                                 window_size_ms=125, overlap_ms=25)
            st.write('#### Señal rectificada')
            # st.line_chart(signalRMS.tolist())
            data = pd.DataFrame({'x': time + delay, 'y': signalRMS})
            plot_signal(data)

            maximo = np.max(signalRMS)
            maximos[muscle] = maximo
            # st.write(f"###### Máximo: {maximo}")

        st.write(maximos)
        st.markdown('#### Selecciona el archivo conventional')
        data2, df2 = load_csv('conventional')
        if data2:
            cont = 0
            for muscle in muscles:
                st.write(f'### {muscle}')
                signal0 = remove_dc_offset(df2[muscle])
                if cont > 0:
                    time0 = df2[f'X[s].{cont}']
                else:
                    time0 = df2['X[s]']
                cont += 4
                sr2 = calculate_sampling_frequency(time0)

                if showOr:
                    st.write('#### Señal original')
                    data = pd.DataFrame({'x': time0, 'y': signal0})
                    plot_signal(data)

                values = st.slider(
                    'Seleccionar rango de muestras',
                    0,
                    len(signal0),
                    (0, len(signal0)),
                    key=f'SLIDER2_{muscle}',
                )

                signal = signal0[values[0]:values[1]]
                time0[values[0]:values[1]]
                delay = time0[values[0]]

                # st.write(f"#### Señal rectificada")
                signalRMS2, time = rms_rectify_signal(signal=signal, sample_rate=sr2,
                                                      window_size_ms=125, overlap_ms=25)
                # data = pd.DataFrame({'x': time+delay,'y': signalRMS2})
                # plot_signal(data)

                st.write('#### Señal normalizada')
                signalNORM = normalize_signal(signalRMS2, maximos[muscle], 100)
                data = pd.DataFrame({'x': time + delay, 'y': signalNORM})
                plot_signal(data, yaxis_title='Signal %')


def process_fft():
    st.markdown('#### Procesando mediante FFT')
    data, df = load_csv('fft')
    if data:
        showOr = st.toggle('¿Mostrar señal original?')
        keys = df.keys()
        muscles = []
        for key in keys:
            if 'EMG' in key:
                muscles.append(key)
        if showOr:
            st.dataframe(df)
            st.write(muscles)

        cont = 0
        for muscle in muscles:
            st.write(f'### {muscle}')
            signal0 = remove_dc_offset(df[muscle])
            if cont > 0:
                time0 = df[f'X[s].{cont}']
            else:
                time0 = df['X[s]']
            cont += 4
            sr = calculate_sampling_frequency(time0)

            if showOr:
                st.write('#### Señal original')
                data = pd.DataFrame({'x': time0, 'y': signal0})
                plot_signal(data)

            values = st.slider(
                'Seleccionar rango de muestras',
                0,
                len(signal0),
                (0, len(signal0)),
                key=f'SLIDER_{muscle}',
            )

            signal = signal0[values[0]:values[1]]
            time2 = time0[values[0]:values[1]]

            st.write('#### Señal recortada')
            data = pd.DataFrame({'x': time2, 'y': signal})
            plot_signal(data)

            N = len(signal)
            freqs, fft_values = compute_fft(signal=signal, sample_rate=sr)
            freqs = freqs[:N // 2][trim:]
            fft_values = np.abs(fft_values)[:N // 2][trim:]

            # df_signal = pd.DataFrame({'Time (s)': df["X[s]"].tolist(), 'Amplitude': signal})
            df_fft = pd.DataFrame({'x': freqs, 'y': fft_values})

            st.write('#### Señal FFT')
            # st.line_chart(df_fft.set_index('Frequency (Hz)'))
            plot_signal(df_fft)
            stats = compute_statistics(freqs[:N // 2][trim:],
                                       fft_values[:N // 2][trim:])
            st.write(stats)


def process_onoff():
    st.markdown('#### Procesando mediante ON-OFF')
    data, df = load_csv('onoff')
    if data:
        showOr = st.toggle('¿Mostrar señal original?')
        keys = df.keys()
        muscles = []
        for key in keys:
            if 'EMG' in key:
                muscles.append(key)
        if showOr:
            st.dataframe(df)
            st.write(muscles)

        cont = 0
        for muscle in muscles:
            st.write(f'### {muscle}')
            signal = remove_dc_offset(df[muscle])
            if cont > 0:
                time0 = df[f'X[s].{cont}']
            else:
                time0 = df['X[s]']
            cont += 4
            sr = calculate_sampling_frequency(time0)
            data0 = pd.DataFrame({'x': time0, 'y': signal})
            if showOr:
                st.write('#### Señal original')
                plot_signal(data0)

            signalRMS0, time = rms_rectify_signal(signal=signal, sample_rate=sr,
                                                  window_size_ms=100, overlap_ms=99)
            signalRMS0 = signalRMS0[trim:]
            time = time[trim:]

            values = st.slider(
                'Seleccionar rango de muestras',
                0,
                len(signalRMS0),
                (0, len(signalRMS0)),
                key=f'ONOFF_SLIDER_{muscle}',
            )

            signalRMS = signalRMS0[values[0]:values[1]]
            time2 = time[values[0]:values[1]]

            st.write('#### Señal rectificada')
            data = pd.DataFrame({'x': time2, 'y': signalRMS.tolist()})
            plot_signal(data)

            threshold = 5 * np.std(signalRMS) + np.average(signalRMS)

            st.write('#### Señal ON/OFF')

            # signalONOFF, time3, bin = check_signal_threshold(time=time, signal=signalRMS0,
            #                                                 threshold=threshold,
            #                                                 window_ms=100)

            bin, time3 = detect_threshold_crossings(time, signalRMS0, threshold,
                                                    window_size_ms=50, overlap_ms=49)

            datar = pd.DataFrame({'x': time, 'y': signalRMS0})
            datao = pd.DataFrame({'x': time3, 'y': bin})
            plot_signal2(
                datar,
                datao,
                threshold,
                x='x',
                y='y',
                x2='x',
                y2='y',
                xt='Time (s)',
                yt='Signal',
                xt2='Time (s)',
                yt2='Signal',
            )


def search_input():
    st.session_state['fecha'] = st.date_input(label='Fecha de búsqueda',
                                              max_value=datetime.today().date())

    if st.toggle('Buscar por RUT'):
        show_data('rut')

    if st.toggle('Buscar por Fecha'):
        show_data('fecha')


def process_input():
    option = st.selectbox(
        r'$\textsf{\large Seleccionar método de análisis}$',
        ('Análisis de máximos', 'Análisis FFT', 'Análisis ON/OFF'),
    )
    if option == 'Análisis de máximos':
        process_max()
    if option == 'Análisis FFT':
        process_fft()
    if option == 'Análisis ON/OFF':
        process_onoff()


def run_git_pull(verbose=False):
    try:
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True,
                                check=True)
        if verbose:
            st.sidebar.success(f'Update successful:\n{result.stdout}')
        else:
            st.sidebar.success('Actualización  👍')
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            st.sidebar.error(f'Update failed:\n{e.stderr}')
        else:
            st.sidebar.error('Actualización  👎')
        return False


def run_pip_install(verbose=False):
    python_path = os.path.join('..', 'env', 'Scripts', 'python')  # Windows
    # python_path = os.path.join('..', 'env', 'bin', 'python')  # Linux/MacOS

    try:
        result = subprocess.run(
            [python_path, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True, text=True, check=True)
        if verbose:
            st.sidebar.success(f'Requirements update successful:\n{result.stdout}')
        else:
            st.sidebar.success('Instalación  👍')
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            st.sidebar.error(f'Requirements update failed:\n{e.stderr}')
        else:
            st.sidebar.error('Instalación  👎')
        return False
