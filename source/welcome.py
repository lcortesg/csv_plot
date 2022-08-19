import streamlit as st


def welcome():
    st.markdown("# Página principal ✨")
    st.sidebar.markdown("# Página Principal ✨")
    st.markdown("## Plataforma de procesamiento de archivos _CSV_ y _TXT_")
    st.markdown(
        "#### Aquí podrás graficar, mezclar, dividir y convertir estos archivos"
    )
    st.markdown(
        "La conversión de archivos TXT a CSV se encuentra en periodo de **prueba**"
    )
    st.markdown(
        """
        #### Funcionalidades
        - **Plot**: Crear gráficos de fuerza en el tiempo, cálculo de máximo, valor medio, y otros parámetros estadísticos.
        - **Split**: Dividir archivos CSV en múltiples exámenes.
        - **Merge**: Mezclar archivos CSV de multiples exámenes en uno.
        - **Convert**: Convertir archivos TXT a CVS.
        - **✨Magia✨**
        """
    )
