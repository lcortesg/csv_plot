import streamlit as st
from PIL import Image
im = Image.open("assets/logos/favicon.png")
st.set_page_config(
    page_title="CSV Handler",
    page_icon=im,
    layout="wide",
)

def welcome():
    st.markdown("# Página principal")
    st.sidebar.markdown("# Página Principal")
    st.markdown("## Plataforma de procesamiento de archivos _CSV_ y _TXT_")
    st.markdown(
        """
        #### Funcionalidades
        - **Plot**: Crear gráficos de fuerza en el tiempo, cálculo de máximo, valor medio, y otros parámetros estadísticos.
        - **Split**: Dividir archivos CSV en múltiples exámenes.
        - **Merge**: Mezclar archivos CSV de multiples exámenes en uno.
        - **Convert**: Convertir archivos TXT a CVS.
        - **Audio**: Comparar datos de audio.
        - **USACH**: Comparar datos de QTM/ABMA.
        - **USACH-MKL**: Comparar datos QTM/ABMA-LITE.
        - **Tobii**: Comparar datos de Tobii Pro.
        """
    )

welcome()
