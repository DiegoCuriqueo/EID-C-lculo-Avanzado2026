# app.py — versión reducida
import streamlit as st
from src.models.distancia import (
    distancia,
    derivadas_parciales,
    diferencial_total,
    comparacion_errores,
)
from src.models.gradiente import (
    gradiente,
    magnitud_gradiente,
    direccion_maximo_crecimiento,
)
from src.utils.state import init_session_state

st.set_page_config(page_title="Propagación de errores - Drones", layout="wide")
init_session_state()
st.title("Análisis de propagación de errores en posicionamiento de drones")

tab1, tab2, tab3 = st.tabs([
    "1. Modelo y datos",
    "2. Derivadas, gradiente y diferencial",
    "4. Visualizaciones y trayectoria",
])

with tab1:
    st.header("Posición del dron y cálculo de la distancia exacta")
    st.caption("El error de medición se ingresa en la Pestaña 2, donde efectivamente se utiliza.")
    st.subheader("Posición (x, y, z)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.posicion["x"] = st.number_input(
            "x (m)", value=st.session_state.posicion["x"]
        )
    with col2:
        st.session_state.posicion["y"] = st.number_input(
            "y (m)", value=st.session_state.posicion["y"]
        )
    with col3:
        st.session_state.posicion["z"] = st.number_input(
            "z (m)", value=st.session_state.posicion["z"]
        )

    x, y, z = st.session_state.posicion.values()
    D = distancia(x, y, z)

    st.subheader("Cálculo de la distancia exacta")
    st.write(
        "La estación base se ubica en el origen (0, 0, 0). La distancia entre "
        "el dron y la estación base se modela con:"
    )
    st.latex(r"D(x,y,z) = \sqrt{x^2 + y^2 + z^2}")
    st.write("Reemplazando la posición ingresada:")
    st.latex(
        rf"D({x:g},\,{y:g},\,{z:g}) = "
        rf"\sqrt{{({x:g})^2 + ({y:g})^2 + ({z:g})^2}} = "
        rf"\sqrt{{{x**2:.3f} + {y**2:.3f} + {z**2:.3f}}} = {D:.4f}\ \text{{m}}"
    )
    st.metric("Distancia exacta D(x,y,z)", f"{D:.4f} m")

with tab2:
    st.header("Derivadas parciales, gradiente y diferencial total")

    from src.models.distancia import (
        distancia,
        derivadas_parciales,
        diferencial_total,
        comparacion_errores,
    )
    from src.models.gradiente import (
        gradiente,
        magnitud_gradiente,
        direccion_maximo_crecimiento,
    )

    # Posición leída desde la Pestaña 1
    x = st.session_state.posicion["x"]
    y = st.session_state.posicion["y"]
    z = st.session_state.posicion["z"]

    st.caption(f"Usando posición actual (Pestaña 1): ({x}, {y}, {z})")

    # --- Error de medición: aquí es donde realmente se utiliza ---
    st.subheader("Error de medición (dx, dy, dz)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.errores["dx"] = st.slider(
            "dx", -2.0, 2.0, st.session_state.errores["dx"]
        )
    with col2:
        st.session_state.errores["dy"] = st.slider(
            "dy", -2.0, 2.0, st.session_state.errores["dy"]
        )
    with col3:
        st.session_state.errores["dz"] = st.slider(
            "dz", -2.0, 2.0, st.session_state.errores["dz"]
        )

    dx = st.session_state.errores["dx"]
    dy = st.session_state.errores["dy"]
    dz = st.session_state.errores["dz"]

    # --- Derivadas parciales ---
    st.subheader("Derivadas parciales")
    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)

    col1, col2, col3 = st.columns(3)
    col1.metric("Respecto a x", f"{dDdx:.4f}")
    col2.metric("Respecto a y", f"{dDdy:.4f}")
    col3.metric("Respecto a z", f"{dDdz:.4f}")

    # --- Gradiente ---
    st.subheader("Vector gradiente")
    grad = gradiente(x, y, z)
    mag = magnitud_gradiente(x, y, z)
    direccion = direccion_maximo_crecimiento(x, y, z)

    st.write(f"Gradiente = [{grad[0]:.4f}, {grad[1]:.4f}, {grad[2]:.4f}]")

    # --- Diferencial total ---
    st.subheader("Diferencial total (estimación del error)")
    dD = diferencial_total(x, y, z, dx, dy, dz)
    D_actual = distancia(x, y, z)

    # Distancia REAL medida, es decir D(x+dx, y+dy, z+dz)
    D_medida = distancia(x + dx, y + dy, z + dz)
    # Aproximación lineal de esa misma distancia usando el diferencial: D + dD
    D_aproximada = D_actual + dD

    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia exacta D(x,y,z)", f"{D_actual:.4f} m")
    col2.metric("Distancia medida D(x+dx,y+dy,z+dz)", f"{D_medida:.4f} m",
                delta=f"{D_medida - D_actual:.4f} m")
    col3.metric("Distancia aproximada (D + dD)", f"{D_aproximada:.4f} m",
                delta=f"{dD:.4f} m")

    with st.expander("Ver fórmula aplicada"):
        st.latex(r"dD = \frac{\partial D}{\partial x}dx + \frac{\partial D}{\partial y}dy + \frac{\partial D}{\partial z}dz")
        st.write(f"dD = ({dDdx:.4f})({dx}) + ({dDdy:.4f})({dy}) + ({dDdz:.4f})({dz}) = {dD:.4f}")
        st.latex(r"D(x+dx,\,y+dy,\,z+dz) \approx D(x,y,z) + dD")
        st.write(f"D_aproximada ≈ {D_actual:.4f} + ({dD:.4f}) = {D_aproximada:.4f} m")

    # --- Comparación error exacto vs. error estimado ---
    st.subheader("Error exacto vs. error estimado por el diferencial")
    comp = comparacion_errores(x, y, z, dx, dy, dz)
    col1, col2, col3 = st.columns(3)
    col1.metric("Error exacto", f"{comp['error_exacto']:.4f} m")
    col2.metric("Error estimado (dD)", f"{comp['error_estimado']:.4f} m")
    col3.metric("Diferencia (residuo)", f"{comp['diferencia_de_error']:.5f} m")

    st.caption("Todos los gráficos y visualizaciones 3D relacionados con esta posición y este error se encuentran en la Pestaña 4.")

with tab3:
    st.header("Visualizaciones 3D")
    st.caption(
        "Aquí se reúnen todas las visualizaciones del modelo: superficie de nivel, "
        "gradiente, plano tangente y el efecto del error sobre una trayectoria hacia un destino."
    )

    from src.models.distancia import distancia
    from src.visualization.plotter_3d import (
        figura_posicion_y_nivel,
        figura_gradiente_y_plano_tangente,
        figura_trayectoria_destino,
    )

    x = st.session_state.posicion["x"]
    y = st.session_state.posicion["y"]
    z = st.session_state.posicion["z"]
    dx = st.session_state.errores["dx"]
    dy = st.session_state.errores["dy"]
    dz = st.session_state.errores["dz"]

    st.info(
        f"Posición real: ({x:g}, {y:g}, {z:g}) m  |  "
        f"Error: (dx={dx:g}, dy={dy:g}, dz={dz:g}) m  "
        "— editables en las Pestañas 1 y 2."
    )

    # --- Gráfico 1: superficie de nivel + posición real/medida ---
    st.subheader("1. Superficie de nivel y posición del dron")
    st.write(
        "Muestra la estación base, la esfera de nivel D(x,y,z) = cte que pasa "
        "por la posición real del dron, y la posición medida (con error)."
    )
    fig1 = figura_posicion_y_nivel(x, y, z, dx, dy, dz)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # --- Gráfico 2: gradiente y plano tangente ---
    st.subheader("2. Gradiente y plano tangente")
    st.write(
        "Muestra la dirección de máximo crecimiento de la distancia (vector "
        "gradiente) y el plano tangente a la superficie de nivel en la posición real."
    )
    fig2 = figura_gradiente_y_plano_tangente(x, y, z)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Gráfico 3: trayectoria hacia un destino ---
    st.subheader("3. Trayectoria hacia un destino: efecto del error de posicionamiento")
    st.write(
        "El dron planea su ruta usando su posición **medida** (con error), pero "
        "parte realmente desde su posición **real**. Esto provoca que, al llegar, "
        "termine en un punto distinto del destino planeado."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.destino["x"] = st.number_input(
            "x destino (m)", value=st.session_state.destino["x"], key="dest_x"
        )
    with col2:
        st.session_state.destino["y"] = st.number_input(
            "y destino (m)", value=st.session_state.destino["y"], key="dest_y"
        )
    with col3:
        st.session_state.destino["z"] = st.number_input(
            "z destino (m)", value=st.session_state.destino["z"], key="dest_z"
        )

    x_dest = st.session_state.destino["x"]
    y_dest = st.session_state.destino["y"]
    z_dest = st.session_state.destino["z"]

    from src.analysis.error_propagation import trayectoria_con_error
    r = trayectoria_con_error(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest)

    col1, col2, col3 = st.columns(3)
    col1.metric("Distancia ideal (real → destino)", f"{r['distancia_ideal']:.3f} m")
    col2.metric("Distancia planeada (medida → destino)", f"{r['distancia_planeada']:.3f} m")
    col3.metric("Desviación final al llegar", f"{r['desviacion']:.3f} m")

    fig3 = figura_trayectoria_destino(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest)
    st.plotly_chart(fig3, use_container_width=True)