# -*- coding: utf-8 -*-
"""
@file    : 8_🪢_Merge.py
@brief   : Handles CSV merging.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

import pandas as pd
import streamlit as st
from PIL import Image
im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)

def csv_merge():

    st.markdown("# CSV Merge 🪢")
    st.sidebar.markdown("# CSV Merge 🪢")

    uploaded_files = st.file_uploader(
        "Elige los archivos CSV para mezclar",
        type=["csv"],
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

        return True

    return False

def main():
    csv_merge()

if __name__ == "__main__":
    main()
