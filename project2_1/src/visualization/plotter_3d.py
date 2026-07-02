import numpy as np
import plotly.graph_objects as go

from src.models.distancia import distancia
from src.models.gradiente import gradiente
from src.analysis.error_propagation import trayectoria_con_error
from src.models.velocidad import velocidad_instantanea

COLOR_BASE = "#1f2937"        # estación base (gris oscuro)
COLOR_SUPERFICIE = "#60a5fa"  # superficie de nivel (celeste translúcido)
COLOR_REAL = "#10b981"        # posición real (verde)
COLOR_MEDIDA = "#ef4444"      # posición medida / con error (rojo)
COLOR_ERROR = "#f59e0b"       # vector de error (naranjo)
COLOR_GRADIENTE = "#8b5cf6"   # vector gradiente (violeta)
COLOR_PLANO = "#facc15"       # plano tangente (amarillo translúcido)
COLOR_DESTINO = "#0ea5e9"     # destino planeado (azul)
COLOR_LLEGADA = "#dc2626"     # punto de llegada real con error (rojo intenso)
COLOR_IDEAL = "#10b981"       # ruta ideal sin error (verde)
COLOR_PLANEADA = "#6b7280"    # ruta "fantasma" que el dron cree seguir (gris)

TEXTFONT = dict(color="black", size=13, family="Arial")

# Tamaños de marcador
MS_BASE = 11
MS_PUNTO = 11
MS_DESTACADO = 10


def _layout_base(fig, titulo):
    """ Aplica un layout consistente y limpio a cualquier figura 3D del proyecto """
    fig.update_layout(
        title=dict(text=titulo, x=0.02, font=dict(size=18)),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            xaxis=dict(color="black", title_font=dict(color="black")),
            yaxis=dict(color="black", title_font=dict(color="black")),
            zaxis=dict(color="black", title_font=dict(color="black")),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.15, y=1.15, z=0.9)),
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.01, x=0.01,
            font=dict(color="black", size=12),
            bgcolor="rgba(255,255,255,0.75)",
        ),
        height=680,
    )
    return fig


def _malla_esfera(radio, resolucion=45):
    """ Genera coordenadas (x, y, z) de una esfera centrada en el origen """
    theta = np.linspace(0, np.pi, resolucion)
    phi = np.linspace(0, 2 * np.pi, resolucion)
    theta, phi = np.meshgrid(theta, phi)
    xs = radio * np.sin(theta) * np.cos(phi)
    ys = radio * np.sin(theta) * np.sin(phi)
    zs = radio * np.cos(theta)
    return xs, ys, zs


def figura_posicion_y_nivel(x, y, z, dx, dy, dz):
    D0 = distancia(x, y, z)
    xm, ym, zm = x + dx, y + dy, z + dz

    fig = go.Figure()

    # --- Superficie de nivel: esfera D(x,y,z) = D0 ---
    if D0 > 0:
        xs, ys, zs = _malla_esfera(D0)
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, COLOR_SUPERFICIE], [1, COLOR_SUPERFICIE]],
            opacity=0.25,
            showscale=False,
            name=f"Superficie de nivel D = {D0:.2f} m",
            hoverinfo="skip",
        ))

    # --- Estación base (origen) ---
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=MS_BASE, color=COLOR_BASE, symbol="diamond"),
        text=["Estación base"], textposition="bottom center", textfont=TEXTFONT,
        name="Estación base",
    ))

    # --- Posición real ---
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_REAL,
                     line=dict(color="black", width=1)),
        text=["Posición real"], textposition="top center", textfont=TEXTFONT,
        name=f"Posición real ({x:.2f}, {y:.2f}, {z:.2f})",
    ))

    # --- Posición medida (con error) ---
    fig.add_trace(go.Scatter3d(
        x=[xm], y=[ym], z=[zm],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_MEDIDA,
                     line=dict(color="black", width=1)),
        text=["Posición medida"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Posición medida ({xm:.2f}, {ym:.2f}, {zm:.2f})",
    ))

    # --- Vector de error más grande y visible (Línea discontinua) ---
    fig.add_trace(go.Scatter3d(
        x=[x, xm], y=[y, ym], z=[z, zm],
        mode="lines",
        line=dict(color=COLOR_ERROR, width=22, dash="dash"), 
        name="Vector de error (dx, dy, dz)",
    ))

    # --- Línea desde la estación base a la posición real (radio) ---
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines",
        line=dict(color=COLOR_BASE, width=8, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))

    return _layout_base(fig, "Superficie de nivel, posición real y posición medida")


def figura_gradiente_y_plano_tangente(x, y, z, escala=None):
    D0 = distancia(x, y, z)
    grad = gradiente(x, y, z)

    if escala is None:
        escala = max(D0 * 0.35, 1.0)

    fig = go.Figure()

    if D0 > 0:
        xs, ys, zs = _malla_esfera(D0)
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, COLOR_SUPERFICIE], [1, COLOR_SUPERFICIE]],
            opacity=0.20,
            showscale=False,
            name=f"Superficie de nivel D = {D0:.2f} m",
            hoverinfo="skip",
        ))

    if D0 > 0:
        n = grad
        aux = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, aux)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)

        s = np.linspace(-escala, escala, 10)
        t = np.linspace(-escala, escala, 10)
        S, T = np.meshgrid(s, t)
        punto = np.array([x, y, z])
        plano = punto[:, None, None] + u[:, None, None] * S + v[:, None, None] * T

        fig.add_trace(go.Surface(
            x=plano[0], y=plano[1], z=plano[2],
            colorscale=[[0, COLOR_PLANO], [1, COLOR_PLANO]],
            opacity=0.45,
            showscale=False,
            showlegend=True,
            name="Plano tangente (linealización)",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_REAL,
                     line=dict(color="black", width=1)),
        text=["Punto (x, y, z)"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Punto ({x:.2f}, {y:.2f}, {z:.2f})",
    ))

    punta = np.array([x, y, z]) + grad * escala
    fig.add_trace(go.Scatter3d(
        x=[x, punta[0]], y=[y, punta[1]], z=[z, punta[2]],
        mode="lines",
        line=dict(color=COLOR_GRADIENTE, width=10),
        name="Vector gradiente",
    ))
    fig.add_trace(go.Cone(
        x=[punta[0]], y=[punta[1]], z=[punta[2]],
        u=[grad[0]], v=[grad[1]], w=[grad[2]],
        sizemode="absolute", sizeref=escala * 0.45,
        anchor="tip", showscale=False,
        colorscale=[[0, COLOR_GRADIENTE], [1, COLOR_GRADIENTE]],
        name="∇D",
    ))

    return _layout_base(fig, "Gradiente y plano tangente en el punto evaluado")


def figura_trayectoria_destino(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest):
    r = trayectoria_con_error(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest)
    real, medida = r["real"], r["medida"]
    destino, llegada = r["destino"], r["punto_llegada"]

    fig = go.Figure()

    # --- Estación base ---
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=MS_BASE, color=COLOR_BASE, symbol="diamond"),
        text=["Estación base"], textposition="bottom center", textfont=TEXTFONT,
        name="Estación base",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, real[0]], y=[0, real[1]], z=[0, real[2]],
        mode="lines", line=dict(color=COLOR_BASE, width=3, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # --- Posición real y medida del dron ---
    fig.add_trace(go.Scatter3d(
        x=[real[0]], y=[real[1]], z=[real[2]],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_REAL,
                     line=dict(color="black", width=1)),
        text=["Dron (posición real)"], textposition="top center", textfont=TEXTFONT,
        name=f"Posición real ({real[0]:.1f}, {real[1]:.1f}, {real[2]:.1f})",
    ))
    fig.add_trace(go.Scatter3d(
        x=[medida[0]], y=[medida[1]], z=[medida[2]],
        mode="markers+text",
        marker=dict(size=MS_PUNTO, color=COLOR_MEDIDA,
                     line=dict(color="black", width=1)),
        text=["Dron (posición medida)"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Posición medida ({medida[0]:.1f}, {medida[1]:.1f}, {medida[2]:.1f})",
    ))

    # --- Destino planeado ---
    fig.add_trace(go.Scatter3d(
        x=[destino[0]], y=[destino[1]], z=[destino[2]],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_DESTINO, symbol="diamond",
                     line=dict(color="black", width=1)),
        text=["Destino"], textposition="top center", textfont=TEXTFONT,
        name=f"Destino ({destino[0]:.1f}, {destino[1]:.1f}, {destino[2]:.1f})",
    ))

    # --- Ruta ideal: real -> destino (sin error) ---
    fig.add_trace(go.Scatter3d(
        x=[real[0], destino[0]], y=[real[1], destino[1]], z=[real[2], destino[2]],
        mode="lines",
        line=dict(color=COLOR_IDEAL, width=8),
        name="Ruta ideal: real → destino (sin error)",
    ))

    # --- Ruta ejecutada: real -> punto de llegada (con error) ---
    fig.add_trace(go.Scatter3d(
        x=[real[0], llegada[0]], y=[real[1], llegada[1]], z=[real[2], llegada[2]],
        mode="lines",
        line=dict(color=COLOR_LLEGADA, width=9, dash="dash"),
        name="Ruta ejecutada: real → llegada (lo que pasa de verdad)",
    ))

    # --- Punto de llegada real ---
    fig.add_trace(go.Scatter3d(
        x=[llegada[0]], y=[llegada[1]], z=[llegada[2]],
        mode="markers+text",
        marker=dict(size=6, color=COLOR_LLEGADA, symbol="x",
                     line=dict(color="black", width=1)),
        text=["Punto de llegada"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Llegada real ({llegada[0]:.1f}, {llegada[1]:.1f}, {llegada[2]:.1f})",
    ))

    fig.add_trace(go.Scatter3d(
        x=[llegada[0], destino[0]], y=[llegada[1], destino[1]], z=[llegada[2], destino[2]],
        mode="lines",
        line=dict(color=COLOR_ERROR, width=24, dash="dot"),
        name=f"Desviación final: {r['desviacion']:.3f} m",
    ))

    return _layout_base(fig, "Trayectoria hacia el destino: efecto del error de posicionamiento")


def figura_trayectoria_velocidad(x1, y1, z1, x2, y2, z2, dt, dx, dy, dz):
    """
    Visualiza la trayectoria real del dron entre t y t+dt, la trayectoria
    calculada a partir de la posición medida (con error) en t+dt, y el
    vector de error entre ambas, ilustrando la propagación del error de
    posición hacia la velocidad estimada (Sección 5 del informe).
    """
    v_real = np.sqrt(
        ((x2 - x1) / dt) ** 2 + ((y2 - y1) / dt) ** 2 + ((z2 - z1) / dt) ** 2
    ) if dt != 0 else 0.0

    x2m, y2m, z2m = x2 + dx, y2 + dy, z2 + dz
    v_medida = velocidad_instantanea(x1, y1, z1, x2m, y2m, z2m, dt)

    fig = go.Figure()

    # --- Posición en el instante t ---
    fig.add_trace(go.Scatter3d(
        x=[x1], y=[y1], z=[z1],
        mode="markers+text",
        marker=dict(size=MS_BASE, color=COLOR_BASE, symbol="diamond",
                     line=dict(color="black", width=1)),
        text=["Posición en t"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Posición en t ({x1:.2f}, {y1:.2f}, {z1:.2f})",
    ))

    # --- Posición real en t+dt ---
    fig.add_trace(go.Scatter3d(
        x=[x2], y=[y2], z=[z2],
        mode="markers+text",
        marker=dict(size=MS_DESTACADO, color=COLOR_REAL,
                     line=dict(color="black", width=1)),
        text=["Posición real en t+Δt"], textposition="top center", textfont=TEXTFONT,
        name=f"Posición real t+Δt ({x2:.2f}, {y2:.2f}, {z2:.2f})",
    ))

    # --- Posición medida (con error) en t+dt ---
    fig.add_trace(go.Scatter3d(
        x=[x2m], y=[y2m], z=[z2m],
        mode="markers+text",
        marker=dict(size=MS_PUNTO, color=COLOR_MEDIDA,
                     line=dict(color="black", width=1)),
        text=["Posición medida en t+Δt"], textposition="bottom center", textfont=TEXTFONT,
        name=f"Posición medida t+Δt ({x2m:.2f}, {y2m:.2f}, {z2m:.2f})",
    ))

    # --- Trayectoria real (vector velocidad real) ---
    fig.add_trace(go.Scatter3d(
        x=[x1, x2], y=[y1, y2], z=[z1, z2],
        mode="lines",
        line=dict(color=COLOR_IDEAL, width=9),
        name=f"Trayectoria real — v = {v_real:.3f} m/s",
    ))

    # --- Trayectoria calculada con la posición medida (vector velocidad medida) ---
    fig.add_trace(go.Scatter3d(
        x=[x1, x2m], y=[y1, y2m], z=[z1, z2m],
        mode="lines",
        line=dict(color=COLOR_MEDIDA, width=9, dash="dash"),
        name=f"Trayectoria medida — v = {v_medida:.3f} m/s",
    ))

    # --- Vector de error entre la posición real y medida en t+dt ---
    fig.add_trace(go.Scatter3d(
        x=[x2, x2m], y=[y2, y2m], z=[z2, z2m],
        mode="lines",
        line=dict(color=COLOR_ERROR, width=18, dash="dot"),
        name="Vector de error (dx, dy, dz)",
    ))

    return _layout_base(fig, "Propagación del error de posición hacia la velocidad")