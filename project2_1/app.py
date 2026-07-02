# app.py — versión reducida
import streamlit as st
from src.models.distancia import (distancia, derivadas_parciales, diferencial_total, comparacion_errores)
from src.models.gradiente import (gradiente, magnitud_gradiente, direccion_maximo_crecimiento)
from src.utils.state import init_session_state
from src.visualization.plotter_3d import (figura_posicion_y_nivel, figura_gradiente_y_plano_tangente, figura_trayectoria_destino, figura_trayectoria_velocidad)
from src.models.velocidad import (velocidad_vectorial, velocidad_instantanea, comparacion_errores_velocidad,analisis_sensibilidad_velocidad)

st.set_page_config(page_title="Propagación de errores - Drones", layout="wide")
init_session_state()
st.title("Análisis de propagación de errores en posicionamiento de drones")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Modelo y datos",
    "2. Derivadas, gradiente y diferencial",
    "3. Velocidad y propagación temporal",
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
    st.header("Extensión del modelo: dependencia temporal y velocidad")
    st.write(
        "Cuando la posición del dron se registra en dos instantes sucesivos "
        "*t* y *t + Δt*, se puede estimar la rapidez instantánea del dron a "
        "partir de esas dos posiciones."
    )
    st.latex(
        r"v \approx \frac{\|(x_2,y_2,z_2) - (x_1,y_1,z_1)\|}{\Delta t} = "
        r"\frac{\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}}{\Delta t}"
    )

    st.subheader("Posiciones en t y en t + Δt")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Posición en t**")
        st.session_state.vel_p1["x"] = st.number_input(
            "x₁ (m)", value=st.session_state.vel_p1["x"], key="vel_x1"
        )
        st.session_state.vel_p1["y"] = st.number_input(
            "y₁ (m)", value=st.session_state.vel_p1["y"], key="vel_y1"
        )
        st.session_state.vel_p1["z"] = st.number_input(
            "z₁ (m)", value=st.session_state.vel_p1["z"], key="vel_z1"
        )
    with col_t2:
        st.markdown("**Posición real en t + Δt**")
        st.session_state.vel_p2["x"] = st.number_input(
            "x₂ (m)", value=st.session_state.vel_p2["x"], key="vel_x2"
        )
        st.session_state.vel_p2["y"] = st.number_input(
            "y₂ (m)", value=st.session_state.vel_p2["y"], key="vel_y2"
        )
        st.session_state.vel_p2["z"] = st.number_input(
            "z₂ (m)", value=st.session_state.vel_p2["z"], key="vel_z2"
        )

    st.session_state.vel_dt = st.number_input(
        "Δt (segundos)", min_value=0.01, value=st.session_state.vel_dt, step=0.1
    )

    x1, y1, z1 = st.session_state.vel_p1.values()
    x2, y2, z2 = st.session_state.vel_p2.values()
    dt = st.session_state.vel_dt

    vx, vy, vz = velocidad_vectorial(x1, y1, z1, x2, y2, z2, dt)
    v = velocidad_instantanea(x1, y1, z1, x2, y2, z2, dt)

    st.subheader("Vector velocidad y rapidez")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("vx", f"{vx:.4f} m/s")
    col2.metric("vy", f"{vy:.4f} m/s")
    col3.metric("vz", f"{vz:.4f} m/s")
    col4.metric("Rapidez v", f"{v:.4f} m/s")

    st.divider()

    st.subheader("Propagación del error de posición hacia la velocidad")
    st.write(
        "El error de medición en la posición registrada en t + Δt también "
        "se propaga hacia el cálculo de la velocidad. La estimación de "
        "primer orden se obtiene aplicando la regla de la cadena:"
    )
    st.latex(
        r"dv \approx \frac{1}{\Delta t}\left("
        r"\frac{v_x}{v}\,dx + \frac{v_y}{v}\,dy + \frac{v_z}{v}\,dz\right)"
    )

    st.markdown("**Error de medición en la posición t + Δt (dx, dy, dz)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.vel_errores["dx"] = st.slider(
            "dx", -2.0, 2.0, st.session_state.vel_errores["dx"], key="vel_dx"
        )
    with col2:
        st.session_state.vel_errores["dy"] = st.slider(
            "dy", -2.0, 2.0, st.session_state.vel_errores["dy"], key="vel_dy"
        )
    with col3:
        st.session_state.vel_errores["dz"] = st.slider(
            "dz", -2.0, 2.0, st.session_state.vel_errores["dz"], key="vel_dz"
        )

    dxv = st.session_state.vel_errores["dx"]
    dyv = st.session_state.vel_errores["dy"]
    dzv = st.session_state.vel_errores["dz"]

    comp_v = comparacion_errores_velocidad(x1, y1, z1, x2, y2, z2, dt, dxv, dyv, dzv)

    if comp_v is None:
        st.warning("Δt no puede ser 0.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Velocidad real", f"{comp_v['v_real']:.4f} m/s")
        col2.metric(
            "Velocidad medida", f"{comp_v['v_medida']:.4f} m/s",
            delta=f"{comp_v['error_exacto']:.4f} m/s",
        )
        col3.metric("Error estimado (dv)", f"{comp_v['error_estimado']:.4f} m/s")

        st.subheader("Error exacto vs. error estimado por el diferencial")
        col1, col2, col3 = st.columns(3)
        col1.metric("Error exacto", f"{comp_v['error_exacto']:.4f} m/s")
        col2.metric("Error estimado", f"{comp_v['error_estimado']:.4f} m/s")
        col3.metric("Diferencia (residuo)", f"{comp_v['diferencia_de_error']:.5f} m/s")

        with st.expander("Ver fórmula aplicada"):
            st.write(
                f"vx = {comp_v['vx']:.4f}, vy = {comp_v['vy']:.4f}, "
                f"vz = {comp_v['vz']:.4f}, v = {comp_v['v_real']:.4f}"
            )
            st.write(
                f"dv ≈ (1/{dt:g}) · "
                f"[({comp_v['vx']:.4f}/{comp_v['v_real']:.4f})·({dxv}) + "
                f"({comp_v['vy']:.4f}/{comp_v['v_real']:.4f})·({dyv}) + "
                f"({comp_v['vz']:.4f}/{comp_v['v_real']:.4f})·({dzv})] "
                f"= {comp_v['error_estimado']:.4f} m/s"
            )

        st.subheader("Análisis de sensibilidad por componente")
        sens_v = analisis_sensibilidad_velocidad(x1, y1, z1, x2, y2, z2, dt, dxv, dyv, dzv)
        col1, col2, col3 = st.columns(3)
        col1.metric("Contribución de x", f"{sens_v['contrib_x_pct']:.1f} %")
        col2.metric("Contribución de y", f"{sens_v['contrib_y_pct']:.1f} %")
        col3.metric("Contribución de z", f"{sens_v['contrib_z_pct']:.1f} %")

        st.subheader("Visualización: trayectoria real vs. trayectoria medida")
        fig_vel = figura_trayectoria_velocidad(x1, y1, z1, x2, y2, z2, dt, dxv, dyv, dzv)
        st.plotly_chart(fig_vel, use_container_width=True)

with tab4:
    st.header("Visualizaciones 3D")
    st.caption(
        "Aquí se reúnen todas las visualizaciones del modelo: superficie de nivel, "
        "gradiente, plano tangente y el efecto del error sobre una trayectoria hacia un destino."
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