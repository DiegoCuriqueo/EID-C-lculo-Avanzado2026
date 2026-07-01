# src/utils/state.py

import streamlit as st

def init_session_state():
    """
    Inicializa todas las variables compartidas en session_state.
    Debe llamarse una sola vez, al inicio de app.py, antes de renderizar las tabs.
    """
    if "posicion" not in st.session_state:
        st.session_state.posicion = {"x": 100.0, "y": 50.0, "z": 30.0}

    if "errores" not in st.session_state:
        st.session_state.errores = {"dx": 0.5, "dy": 0.3, "dz": 0.2}

    if "modo_error" not in st.session_state:
        st.session_state.modo_error = "manual"  # o "estadistico"

    if "sigma" not in st.session_state:
        st.session_state.sigma = {"x": 0.5, "y": 0.5, "z": 0.5}

    if "n_simulaciones" not in st.session_state:
        st.session_state.n_simulaciones = 1000

    if "df_ruta" not in st.session_state:
        st.session_state.df_ruta = None  # se llena si cargan el CSV

    if "destino" not in st.session_state:
        # Punto de destino planeado para la visualización de trayectoria (Pestaña 4)
        st.session_state.destino = {"x": 300.0, "y": 300.0, "z": 150.0}