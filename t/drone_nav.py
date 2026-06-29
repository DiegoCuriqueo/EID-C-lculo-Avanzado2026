"""
drone_nav.py
Lógica principal para el análisis de propagación de errores
en sistemas de posicionamiento para drones.
MATE1189 - Cálculo Avanzado - Proyecto Final 8
"""

import numpy as np


# ──────────────────────────────────────────────
# 1. MODELO DE DISTANCIA
# ──────────────────────────────────────────────

def distancia(x: float, y: float, z: float) -> float:
    """D(x,y,z) = sqrt(x² + y² + z²)"""
    return np.sqrt(x**2 + y**2 + z**2)


def distancia_exacta_con_error(x, y, z, dx, dy, dz) -> float:
    """Distancia exacta evaluada en la posición medida (x+dx, y+dy, z+dz)."""
    return distancia(x + dx, y + dy, z + dz)


# ──────────────────────────────────────────────
# 2. DERIVADAS PARCIALES Y GRADIENTE
# ──────────────────────────────────────────────

def derivadas_parciales(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Retorna (∂D/∂x, ∂D/∂y, ∂D/∂z) evaluadas en (x, y, z).
    ∂D/∂x = x / D,  análogamente para y, z.
    """
    D = distancia(x, y, z)
    if D == 0:
        return (0.0, 0.0, 0.0)
    return (x / D, y / D, z / D)


def gradiente(x: float, y: float, z: float) -> np.ndarray:
    """∇D(x,y,z) como vector numpy."""
    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)
    return np.array([dDdx, dDdy, dDdz])


def magnitud_gradiente(x: float, y: float, z: float) -> float:
    """||∇D|| — siempre 1 para la función distancia al origen (cuando D≠0)."""
    return float(np.linalg.norm(gradiente(x, y, z)))


# ──────────────────────────────────────────────
# 3. DIFERENCIAL TOTAL Y APROXIMACIÓN LINEAL
# ──────────────────────────────────────────────

def diferencial_total(x, y, z, dx, dy, dz) -> float:
    """
    dD ≈ (∂D/∂x)dx + (∂D/∂y)dy + (∂D/∂z)dz
    Estimación lineal del error propagado.
    """
    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)
    return dDdx * dx + dDdy * dy + dDdz * dz


def error_exacto(x, y, z, dx, dy, dz) -> float:
    """ΔD = D(x+dx, y+dy, z+dz) - D(x, y, z)"""
    return distancia_exacta_con_error(x, y, z, dx, dy, dz) - distancia(x, y, z)


def error_relativo_diferencial(x, y, z, dx, dy, dz) -> dict:
    """
    Compara el diferencial dD con el error exacto ΔD.
    Retorna un dict con todas las magnitudes relevantes.
    """
    D0 = distancia(x, y, z)
    dD = diferencial_total(x, y, z, dx, dy, dz)
    delta_D = error_exacto(x, y, z, dx, dy, dz)
    error_aprox = abs(dD - delta_D)

    return {
        "D0": D0,
        "D_medida": distancia_exacta_con_error(x, y, z, dx, dy, dz),
        "dD": dD,
        "delta_D": delta_D,
        "error_aprox": error_aprox,
        "error_relativo_dD": abs(dD / D0) * 100 if D0 != 0 else None,
        "error_relativo_exacto": abs(delta_D / D0) * 100 if D0 != 0 else None,
    }


# ──────────────────────────────────────────────
# 4. ANÁLISIS DE SENSIBILIDAD
# ──────────────────────────────────────────────

def contribucion_por_coordenada(x, y, z, dx, dy, dz) -> dict:
    """
    Contribución de cada coordenada al diferencial total (en valor absoluto).
    Útil para identificar qué eje domina el error.
    """
    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)
    cx = abs(dDdx * dx)
    cy = abs(dDdy * dy)
    cz = abs(dDdz * dz)
    total = cx + cy + cz
    return {
        "cx": cx, "cy": cy, "cz": cz,
        "total": total,
        "pct_x": 100 * cx / total if total != 0 else 0,
        "pct_y": 100 * cy / total if total != 0 else 0,
        "pct_z": 100 * cz / total if total != 0 else 0,
    }


def analisis_sensibilidad_barrido(x, y, z, rango_error=0.5, pasos=50) -> dict:
    """
    Varía cada error individualmente de 0 a rango_error
    manteniendo los otros en 0. Retorna arrays para graficar.
    """
    errores = np.linspace(0, rango_error, pasos)
    dD_dx_var = [diferencial_total(x, y, z, e, 0, 0) for e in errores]
    dD_dy_var = [diferencial_total(x, y, z, 0, e, 0) for e in errores]
    dD_dz_var = [diferencial_total(x, y, z, 0, 0, e) for e in errores]
    return {"errores": errores, "dD_x": dD_dx_var, "dD_y": dD_dy_var, "dD_z": dD_dz_var}


# ──────────────────────────────────────────────
# 5. EXTENSIÓN CON TIEMPO (modelo cinemático)
# ──────────────────────────────────────────────

def posicion_tiempo(t: float, x0, y0, z0, vx, vy, vz) -> tuple[float, float, float]:
    """
    Modelo de posición lineal: r(t) = r0 + v·t
    (x(t), y(t), z(t)) = (x0 + vx·t, y0 + vy·t, z0 + vz·t)
    """
    return (x0 + vx * t, y0 + vy * t, z0 + vz * t)


def velocidad_escalar(vx: float, vy: float, vz: float) -> float:
    """||v|| = sqrt(vx² + vy² + vz²)"""
    return np.sqrt(vx**2 + vy**2 + vz**2)


def error_velocidad_diferencial(vx, vy, vz, dvx, dvy, dvz) -> dict:
    """
    Propagación de errores en la velocidad escalar.
    v = sqrt(vx² + vy² + vz²)
    dv = (vx·dvx + vy·dvy + vz·dvz) / v
    """
    v = velocidad_escalar(vx, vy, vz)
    if v == 0:
        return {"v": 0, "dv": 0, "delta_v_exacto": 0}
    dv = (vx * dvx + vy * dvy + vz * dvz) / v
    v_med = velocidad_escalar(vx + dvx, vy + dvy, vz + dvz)
    return {
        "v": v,
        "v_medida": v_med,
        "dv": dv,
        "delta_v_exacto": v_med - v,
        "error_aprox": abs(dv - (v_med - v)),
    }


def distancia_tiempo(t, x0, y0, z0, vx, vy, vz) -> float:
    """D(t) = ||r(t)||"""
    x, y, z = posicion_tiempo(t, x0, y0, z0, vx, vy, vz)
    return distancia(x, y, z)


def error_distancia_con_tiempo(t, x0, y0, z0, vx, vy, vz,
                                dx0, dy0, dz0) -> dict:
    """
    Propaga errores en posición inicial hacia D(t).
    Los errores dx0, dy0, dz0 se trasladan directamente a dx, dy, dz en tiempo t.
    """
    x, y, z = posicion_tiempo(t, x0, y0, z0, vx, vy, vz)
    return error_relativo_diferencial(x, y, z, dx0, dy0, dz0)


# ──────────────────────────────────────────────
# 6. SIMULACIÓN MONTECARLO
# ──────────────────────────────────────────────

def simulacion_montecarlo(x, y, z, sigma_x, sigma_y, sigma_z,
                           n_muestras=10_000, semilla=42) -> dict:
    """
    Genera n_muestras de errores gaussianos y calcula la distribución
    del error en la distancia. Útil para validar la aproximación lineal.
    """
    rng = np.random.default_rng(semilla)
    dx_s = rng.normal(0, sigma_x, n_muestras)
    dy_s = rng.normal(0, sigma_y, n_muestras)
    dz_s = rng.normal(0, sigma_z, n_muestras)

    D0 = distancia(x, y, z)
    delta_D_exacto = distancia(x + dx_s, y + dy_s, z + dz_s) - D0

    dDdx, dDdy, dDdz = derivadas_parciales(x, y, z)
    dD_lineal = dDdx * dx_s + dDdy * dy_s + dDdz * dz_s

    return {
        "delta_D_exacto": delta_D_exacto,
        "dD_lineal": dD_lineal,
        "media_exacto": float(np.mean(delta_D_exacto)),
        "std_exacto": float(np.std(delta_D_exacto)),
        "media_lineal": float(np.mean(dD_lineal)),
        "std_lineal": float(np.std(dD_lineal)),
        "D0": D0,
    }