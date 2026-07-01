# app.py — versión reducida
import streamlit as st
from src.models import distancia, gradiente, velocidad
from src.analysis import error_propagation, sensitivity, comparison
from src.visualization import map_plotter, error_plots
from src.data_processing import loader, errors
from src.utils.state import init_session_state

st.set_page_config(page_title="Propagación de errores - Drones", layout="wide")
init_session_state()
st.title("Análisis de propagación de errores en posicionamiento de drones")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Modelo y datos",
    "2. Derivadas, gradiente y diferencial",
    "3. Comparación y sensibilidad",
    "4. Simulación y extensión temporal",
])

with tab1:
    st.header("Posición del dron, errores y superficies de nivel")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posición (x, y, z)")
        st.session_state.posicion["x"] = st.number_input(
            "x (m)", value=st.session_state.posicion["x"]
        )
        st.session_state.posicion["y"] = st.number_input(
            "y (m)", value=st.session_state.posicion["y"]
        )
        st.session_state.posicion["z"] = st.number_input(
            "z (m)", value=st.session_state.posicion["z"]
        )

    with col2:
        st.subheader("Error de medición (dx, dy, dz)")
        st.session_state.errores["dx"] = st.slider(
            "dx", -2.0, 2.0, st.session_state.errores["dx"]
        )
        st.session_state.errores["dy"] = st.slider(
            "dy", -2.0, 2.0, st.session_state.errores["dy"]
        )
        st.session_state.errores["dz"] = st.slider(
            "dz", -2.0, 2.0, st.session_state.errores["dz"]
        )

    # Aquí ya se usa una función real de distancia.py
    from src.models.distancia import distancia
    x, y, z = st.session_state.posicion.values()
    D = distancia(x, y, z)
    st.metric("Distancia exacta D(x,y,z)", f"{D:.3f} m")

with tab2:
    st.header("Derivadas parciales, gradiente y diferencial total")

    from src.models.distancia import distancia, derivadas_parciales, diferencial_total
    from src.models.gradiente import (
        gradiente,
        magnitud_gradiente,
        direccion_maximo_crecimiento,
        perpendicularidad_gradiente_nivel,
    )

    # Leer del estado compartido (ya cargado en Pestaña 1)
    x = st.session_state.posicion["x"]
    y = st.session_state.posicion["y"]
    z = st.session_state.posicion["z"]
    dx = st.session_state.errores["dx"]
    dy = st.session_state.errores["dy"]
    dz = st.session_state.errores["dz"]

    st.caption(f"Usando posición actual: ({x}, {y}, {z})  |  Errores: ({dx}, {dy}, {dz})")

    # --- Derivadas parciales ---
    st.subheader("Derivadas parciales")
    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)

    col1, col2, col3 = st.columns(3)
    col1.metric("∂D/∂x", f"{dDdx:.4f}")
    col2.metric("∂D/∂y", f"{dDdy:.4f}")
    col3.metric("∂D/∂z", f"{dDdz:.4f}")

    # --- Gradiente ---
    st.subheader("Vector gradiente ∇D")
    grad = gradiente(x, y, z)
    mag = magnitud_gradiente(x, y, z)
    direccion = direccion_maximo_crecimiento(x, y, z)

    st.write(f"∇D(x,y,z) = [{grad[0]:.4f}, {grad[1]:.4f}, {grad[2]:.4f}]")
    st.metric("Magnitud del gradiente", f"{mag:.6f}")

    if abs(mag - 1.0) < 1e-6:
        st.success("✅ El gradiente es unitario, como se espera para D(x,y,z) = √(x²+y²+z²)")
    else:
        st.error(f"⚠️ El gradiente debería tener magnitud 1, pero dio {mag:.6f}. Revisar implementación.")

    st.write(f"Dirección de máximo crecimiento: [{direccion[0]:.4f}, {direccion[1]:.4f}, {direccion[2]:.4f}]")

    # --- Perpendicularidad con la superficie de nivel ---
    st.subheader("Relación con la superficie de nivel")
    perp = perpendicularidad_gradiente_nivel(x, y, z)
    st.json(perp)

    # --- Diferencial total ---
    st.subheader("Diferencial total (estimación del error)")
    dD = diferencial_total(x, y, z, dx, dy, dz)
    D_actual = distancia(x, y, z)

    col1, col2 = st.columns(2)
    col1.metric("Distancia exacta D(x,y,z)", f"{D_actual:.4f} m")
    col2.metric("Error estimado (dD)", f"{dD:.4f} m")

    with st.expander("Ver fórmula aplicada"):
        st.latex(r"dD = \frac{\partial D}{\partial x}dx + \frac{\partial D}{\partial y}dy + \frac{\partial D}{\partial z}dz")
        st.write(f"dD = ({dDdx:.4f})({dx}) + ({dDdy:.4f})({dy}) + ({dDdz:.4f})({dz}) = {dD:.4f}")


with tab3:
    st.header("Comparación exacto vs. aproximado, y sensibilidad")
    # comparison.comparar_con_exacto(...)
    # sensitivity.analizar(...)

with tab4:
    st.header("Simulación de escenarios y extensión con velocidad")
    # velocidad.calcular_velocidad(...)
    # simulación con múltiples (dx, dy, dz)