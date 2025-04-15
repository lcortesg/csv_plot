"""
@file    : 9_⚡️_Split.py
@brief   : Handles CSV splitting.
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

def csv_split(length=3138):

    st.markdown("# CSV Split ⚡️")
    st.sidebar.markdown("# CSV Split ⚡️")

    uploaded_files = st.file_uploader(
        "Elige los archivos CSV para dividir",
        type=["csv"],
        accept_multiple_files=True,
        help="Selecciona uno o más archivos CSV para dividir",
    )

    if len(uploaded_files) > 0:

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

        return True

    return False

def main():
    csv_split()

if __name__ == "__main__":
    main()
