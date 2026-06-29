"""
app.py
Interfaz Streamlit — Análisis de propagación de errores en sistemas de
posicionamiento para drones.
MATE1189 - Cálculo Avanzado - Proyecto Final 8
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import drone_nav as dn

# ──────────────────────────────────────────────
# CONFIG GENERAL
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Propagación de Errores — Drones",
    page_icon="🛸",
    layout="wide",
)

st.title("🛸 Análisis de Propagación de Errores en Sistemas de Posicionamiento para Drones")
st.caption("MATE1189 · Cálculo Avanzado · Proyecto Final 8")

tabs = st.tabs([
    "📐 Modelo Base",
    "📊 Sensibilidad",
    "🕐 Extensión Temporal",
    "🎲 Simulación Monte Carlo",
])

# ──────────────────────────────────────────────
# TAB 1 — MODELO BASE
# ──────────────────────────────────────────────
with tabs[0]:
    st.header("Modelo Base: D(x, y, z) = √(x² + y² + z²)")

    st.markdown(
        r"""
        Dado que la estación base se ubica en el origen, la distancia al dron ubicado en
        $(x, y, z)$ es:
        $$D(x,y,z) = \sqrt{x^2 + y^2 + z^2}$$

        Si los sensores miden $(x+dx,\, y+dy,\, z+dz)$, el error propagado se aproxima con
        el diferencial total:
        $$dD = \frac{\partial D}{\partial x}dx + \frac{\partial D}{\partial y}dy + \frac{\partial D}{\partial z}dz
             = \frac{x\,dx + y\,dy + z\,dz}{D}$$
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posición del dron (m)")
        x = st.number_input("x", value=30.0, step=1.0, key="x_base")
        y = st.number_input("y", value=40.0, step=1.0, key="y_base")
        z = st.number_input("z", value=50.0, step=1.0, key="z_base")

    with col2:
        st.subheader("Errores de medición (m)")
        dx = st.number_input("dx", value=0.5, step=0.1, format="%.3f", key="dx")
        dy = st.number_input("dy", value=-0.3, step=0.1, format="%.3f", key="dy")
        dz = st.number_input("dz", value=0.2, step=0.1, format="%.3f", key="dz")

    # ── Cálculos ──
    res = dn.error_relativo_diferencial(x, y, z, dx, dy, dz)
    grad = dn.gradiente(x, y, z)
    contrib = dn.contribucion_por_coordenada(x, y, z, dx, dy, dz)
    dDdx, dDdy, dDdz = dn.derivadas_parciales(x, y, z)

    st.divider()

    # ── Métricas principales ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("D₀ (distancia real)", f"{res['D0']:.4f} m")
    m2.metric("D medida", f"{res['D_medida']:.4f} m")
    m3.metric("dD (diferencial)", f"{res['dD']:.4f} m")
    m4.metric("ΔD (error exacto)", f"{res['delta_D']:.4f} m")

    e1, e2, e3 = st.columns(3)
    e1.metric("Error de aprox. |dD − ΔD|", f"{res['error_aprox']:.6f} m")
    e2.metric("Error relativo dD", f"{res['error_relativo_dD']:.4f} %" if res['error_relativo_dD'] else "—")
    e3.metric("Error relativo exacto", f"{res['error_relativo_exacto']:.4f} %" if res['error_relativo_exacto'] else "—")

    st.divider()
    col_g, col_d = st.columns(2)

    with col_g:
        st.subheader("Derivadas parciales y Gradiente")
        st.markdown(
            rf"""
            | Variable | Derivada parcial | Valor |
            |----------|-----------------|-------|
            | $x$ | $\partial D/\partial x = x/D$ | `{dDdx:.6f}` |
            | $y$ | $\partial D/\partial y = y/D$ | `{dDdy:.6f}` |
            | $z$ | $\partial D/\partial z = z/D$ | `{dDdz:.6f}` |

            **Gradiente:** $\nabla D = ({grad[0]:.4f},\; {grad[1]:.4f},\; {grad[2]:.4f})$

            $\|\nabla D\| = {np.linalg.norm(grad):.6f}$ *(siempre = 1 para D ≠ 0)*
            """
        )

    with col_d:
        st.subheader("Contribución de cada coordenada al error")
        fig_contrib, ax_c = plt.subplots(figsize=(4, 3))
        labels = ["dx (x)", "dy (y)", "dz (z)"]
        values = [contrib["pct_x"], contrib["pct_y"], contrib["pct_z"]]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        ax_c.bar(labels, values, color=colors)
        ax_c.set_ylabel("Contribución (%)")
        ax_c.set_title("Porcentaje del diferencial total")
        ax_c.set_ylim(0, 110)
        for i, v in enumerate(values):
            ax_c.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        st.pyplot(fig_contrib, use_container_width=True)

    # ── Visualización 3D ──
    st.subheader("Visualización geométrica")
    fig3d = plt.figure(figsize=(7, 5))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Superficie de nivel (esfera)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    D0 = res["D0"]
    xs = D0 * np.outer(np.cos(u), np.sin(v))
    ys = D0 * np.outer(np.sin(u), np.sin(v))
    zs = D0 * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(xs, ys, zs, alpha=0.12, color="steelblue")

    # Posición real y medida
    ax3d.scatter([x], [y], [z], color="green", s=80, zorder=5, label="Posición real")
    ax3d.scatter([x + dx], [y + dy], [z + dz], color="red", s=80, zorder=5, label="Posición medida")

    # Gradiente (normalizado)
    g = grad * 5
    ax3d.quiver(x, y, z, g[0], g[1], g[2], color="orange", linewidth=2, label="∇D")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Superficie de nivel D = D₀ y posiciones")
    ax3d.legend(fontsize=8)
    st.pyplot(fig3d, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 2 — SENSIBILIDAD
# ──────────────────────────────────────────────
with tabs[1]:
    st.header("Análisis de Sensibilidad")
    st.markdown(
        r"""
        Se varía cada error de coordenada de manera independiente (manteniendo los otros en 0)
        para observar cómo cambia el diferencial total $dD$.
        """
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        xs_in = st.number_input("x", value=30.0, step=1.0, key="xs_x")
        ys_in = st.number_input("y", value=40.0, step=1.0, key="xs_y")
        zs_in = st.number_input("z", value=50.0, step=1.0, key="xs_z")
    with col_s2:
        rango = st.slider("Rango máximo de error (m)", 0.1, 5.0, 1.0, 0.1)
        pasos = st.slider("Puntos de barrido", 20, 200, 80)

    datos = dn.analisis_sensibilidad_barrido(xs_in, ys_in, zs_in, rango, pasos)
    errores = datos["errores"]

    fig_sens, ax_s = plt.subplots(figsize=(8, 4))
    ax_s.plot(errores, datos["dD_x"], label="Varía dx (dy=dz=0)", color="#4C72B0")
    ax_s.plot(errores, datos["dD_y"], label="Varía dy (dx=dz=0)", color="#DD8452")
    ax_s.plot(errores, datos["dD_z"], label="Varía dz (dx=dy=0)", color="#55A868")
    ax_s.set_xlabel("Error en coordenada (m)")
    ax_s.set_ylabel("dD (m)")
    ax_s.set_title("Sensibilidad de dD ante cada coordenada")
    ax_s.legend()
    ax_s.grid(alpha=0.3)
    st.pyplot(fig_sens, use_container_width=True)

    st.markdown(
        r"""
        **Interpretación:** la pendiente de cada curva equivale a la derivada parcial correspondiente
        $\partial D / \partial x_i$, que a su vez es el coseno director del vector posición.
        La coordenada con mayor derivada parcial domina el error propagado.
        """
    )

    # Tabla de derivadas parciales en la posición elegida
    dp = dn.derivadas_parciales(xs_in, ys_in, zs_in)
    st.markdown(
        f"""
        | Derivada parcial | Valor en ({xs_in}, {ys_in}, {zs_in}) |
        |-----------------|--------------------------------------|
        | ∂D/∂x | `{dp[0]:.6f}` |
        | ∂D/∂y | `{dp[1]:.6f}` |
        | ∂D/∂z | `{dp[2]:.6f}` |
        """
    )

# ──────────────────────────────────────────────
# TAB 3 — EXTENSIÓN TEMPORAL
# ──────────────────────────────────────────────
with tabs[2]:
    st.header("Extensión Temporal: r(t) = r₀ + v·t")
    st.markdown(
        r"""
        El dron se mueve con velocidad constante. La posición en el instante $t$ es:
        $$\mathbf{r}(t) = (x_0 + v_x t,\; y_0 + v_y t,\; z_0 + v_z t)$$

        El error en la posición inicial $(\delta x_0, \delta y_0, \delta z_0)$ se propaga
        directamente a $\mathbf{r}(t)$, por lo que el análisis diferencial previo aplica
        evaluado en $\mathbf{r}(t)$.

        Para la **velocidad escalar** $v = \|\mathbf{v}\|$, el diferencial es:
        $$dv = \frac{v_x\,dv_x + v_y\,dv_y + v_z\,dv_z}{v}$$
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Posición inicial (m)")
        x0 = st.number_input("x₀", value=10.0, key="x0")
        y0 = st.number_input("y₀", value=20.0, key="y0")
        z0 = st.number_input("z₀", value=30.0, key="z0")
        st.subheader("Errores en posición inicial (m)")
        dx0 = st.number_input("δx₀", value=0.3, format="%.3f", key="dx0")
        dy0 = st.number_input("δy₀", value=0.2, format="%.3f", key="dy0")
        dz0 = st.number_input("δz₀", value=0.1, format="%.3f", key="dz0")

    with c2:
        st.subheader("Velocidad (m/s)")
        vx = st.number_input("vx", value=5.0, key="vx")
        vy = st.number_input("vy", value=3.0, key="vy")
        vz = st.number_input("vz", value=2.0, key="vz")
        st.subheader("Errores en velocidad (m/s)")
        dvx = st.number_input("δvx", value=0.05, format="%.3f", key="dvx")
        dvy = st.number_input("δvy", value=0.03, format="%.3f", key="dvy")
        dvz = st.number_input("δvz", value=0.02, format="%.3f", key="dvz")

    t_max = st.slider("Tiempo máximo (s)", 1.0, 60.0, 20.0, 1.0)
    tiempos = np.linspace(0, t_max, 200)

    # Arrays para graficar
    D_t = [dn.distancia_tiempo(t, x0, y0, z0, vx, vy, vz) for t in tiempos]
    dD_t = [dn.error_distancia_con_tiempo(t, x0, y0, z0, vx, vy, vz, dx0, dy0, dz0)["dD"]
            for t in tiempos]
    dD_exact = [dn.error_distancia_con_tiempo(t, x0, y0, z0, vx, vy, vz, dx0, dy0, dz0)["delta_D"]
                for t in tiempos]

    fig_t, axes_t = plt.subplots(1, 2, figsize=(12, 4))

    axes_t[0].plot(tiempos, D_t, color="steelblue")
    axes_t[0].set_xlabel("t (s)")
    axes_t[0].set_ylabel("D(t) (m)")
    axes_t[0].set_title("Distancia al origen vs tiempo")
    axes_t[0].grid(alpha=0.3)

    axes_t[1].plot(tiempos, dD_t, label="dD (diferencial)", color="orange")
    axes_t[1].plot(tiempos, dD_exact, label="ΔD (exacto)", linestyle="--", color="red")
    axes_t[1].set_xlabel("t (s)")
    axes_t[1].set_ylabel("Error en D (m)")
    axes_t[1].set_title("Propagación del error en D(t)")
    axes_t[1].legend()
    axes_t[1].grid(alpha=0.3)

    st.pyplot(fig_t, use_container_width=True)

    # Error en velocidad
    st.divider()
    st.subheader("Error propagado en la velocidad escalar")
    res_v = dn.error_velocidad_diferencial(vx, vy, vz, dvx, dvy, dvz)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("v (m/s)", f"{res_v['v']:.4f}")
    v2.metric("v medida (m/s)", f"{res_v['v_medida']:.4f}")
    v3.metric("dv (diferencial)", f"{res_v['dv']:.5f}")
    v4.metric("|dv − Δv|", f"{res_v['error_aprox']:.7f}")

# ──────────────────────────────────────────────
# TAB 4 — SIMULACIÓN MONTE CARLO
# ──────────────────────────────────────────────
with tabs[3]:
    st.header("Simulación Monte Carlo")
    st.markdown(
        r"""
        Se generan $N$ realizaciones de errores gaussianos
        $(\delta x, \delta y, \delta z) \sim \mathcal{N}(0, \sigma_i)$
        para estimar la distribución del error en $D$ y comparar con la aproximación lineal.
        """
    )

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.subheader("Posición de referencia (m)")
        xmc = st.number_input("x", value=30.0, key="xmc")
        ymc = st.number_input("y", value=40.0, key="ymc")
        zmc = st.number_input("z", value=50.0, key="zmc")
    with col_mc2:
        st.subheader("Desviaciones estándar (m)")
        sx = st.number_input("σx", value=0.3, min_value=0.01, format="%.3f", key="sx")
        sy = st.number_input("σy", value=0.3, min_value=0.01, format="%.3f", key="sy")
        sz = st.number_input("σz", value=0.3, min_value=0.01, format="%.3f", key="sz")
        n_samples = st.select_slider("N muestras", [1000, 5000, 10000, 50000], value=10000)

    if st.button("▶ Ejecutar simulación"):
        mc = dn.simulacion_montecarlo(xmc, ymc, zmc, sx, sy, sz, n_samples)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("D₀ (m)", f"{mc['D0']:.4f}")
        mc2.metric("μ(ΔD) exacto", f"{mc['media_exacto']:.5f}")
        mc3.metric("σ(ΔD) exacto", f"{mc['std_exacto']:.5f}")
        mc4.metric("σ(dD) lineal", f"{mc['std_lineal']:.5f}")

        fig_mc, ax_mc = plt.subplots(figsize=(9, 4))
        bins = min(80, int(np.sqrt(n_samples)))
        ax_mc.hist(mc["delta_D_exacto"], bins=bins, alpha=0.6,
                   label=f"ΔD exacto (σ={mc['std_exacto']:.4f})", color="steelblue", density=True)
        ax_mc.hist(mc["dD_lineal"], bins=bins, alpha=0.6,
                   label=f"dD lineal (σ={mc['std_lineal']:.4f})", color="orange", density=True)
        ax_mc.set_xlabel("Error en D (m)")
        ax_mc.set_ylabel("Densidad")
        ax_mc.set_title("Distribución del error en la distancia (Monte Carlo)")
        ax_mc.legend()
        ax_mc.grid(alpha=0.3)
        st.pyplot(fig_mc, use_container_width=True)

        st.markdown(
            r"""
            **Interpretación:** cuando los errores son pequeños en relación a la distancia $D_0$,
            las distribuciones del error exacto $\Delta D$ y el diferencial $dD$ se superponen casi
            perfectamente, validando la aproximación lineal. La diferencia crece conforme aumentan
            $\sigma_i / D_0$.
            """
        )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption("MATE1189 · Cálculo Avanzado · Proyecto Final 8 · Dr. Vidal Yunge · 2026-1")