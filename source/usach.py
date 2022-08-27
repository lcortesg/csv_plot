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
from io import BytesIO
import csv
import pandas as pd
import streamlit as st


def usach_plot():

    st.markdown("# TXT Convert ❄️")
    st.sidebar.markdown("# TXT Convert ❄️")

    uploaded_files = st.file_uploader(
        "Hola Nicco! elige los archivos TXT para convertir",
        type=["txt"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos TXT para convertir",
    )

    if len(uploaded_files) > 0:

        
        for uploaded_file in uploaded_files:
            data = uploaded_file.read()
            name = uploaded_file.name
            format = name[-3:]
            new_line = ""
            file = str(data, "utf-8").split("\n")
            word_list = file[1].split("\t")
            word_list[0] = "XY"
            new_line = ",".join(word_list)
            file.pop(0)
            file[0] = new_line
            filetxt = ""

            for line in file:
                filetxt = filetxt + line.replace("\t", ",")

            st.download_button(
                label=f'Descargar {name.replace(format,"csv")}',
                data=filetxt,
                file_name=name.replace(format, "csv"),
                mime="text/csv",
            )

            if st.checkbox(f'Graficar {name.replace(format,"csv")}'):
                buffer = io.StringIO(filetxt)
                df = pd.read_csv(filepath_or_buffer = buffer)
                keys = df.keys()
                df = df.set_index('XY')
                for key in keys:
                    if "Unnamed" in key:
                        df = df.drop(f'{key}', axis=1)
                st.line_chart(df)

        return True

    return False
