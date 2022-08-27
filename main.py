# -*- coding: utf-8 -*-
"""
@file     : CSV Handler
@brief   : Handles CSV and TXT file conversion and plotting.
@date    : 2022/08/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl
@bug     : None.
"""

from PIL import Image
import streamlit as st
from source.plot import csv_plot
from source.split import csv_split
from source.merge import csv_merge
from source.convert import csv_convert
from source.welcome import welcome
from source.audio import wav_plot
from source.usach import usach_plot


def main():

    im = Image.open("assets/logos/favicon.png")

    st.set_page_config(
        page_title="CSV Handler",
        page_icon=im,
        layout="wide",
    )

    functions = {
        "Principal": welcome,
        "Plot": csv_plot,
        "Split": csv_split,
        "Merge": csv_merge,
        "Convert": csv_convert,
        "Audio": wav_plot,
        "USACH": usach_plot,
    }

    selected_function = st.sidebar.selectbox(
        "Hola Nicco! ¿Que quieres hacer hoy?", functions.keys()
    )

    if functions[selected_function]():
        st.markdown("#### ¡Proceso finalizado con éxito! 🥳🎉🎊🎈")


if __name__ == "__main__":
    main()
