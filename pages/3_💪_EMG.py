# -*- coding: utf-8 -*-
"""
@file    : 3_📈_Procesar_EMG.py
@brief   : Handles EMG processing
@date    : 2024/08/22
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import streamlit as st

from source.functions import clear_page, process_input

clear_page('Procesar')


def main():
    st.markdown('# 📈 Procesar Examen EMG')
    process_input()


if __name__ == '__main__':
    main()
