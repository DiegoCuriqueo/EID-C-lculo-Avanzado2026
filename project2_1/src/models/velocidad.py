"""
Módulo para el cálculo de velocidad y propagación de errores en velocidad.
"""
import numpy as np
import pandas as pd

def velocidad_instantanea(x1, y1, z1, x2, y2, z2, dt):
    """
    Calcula la rapidez instantánea entre dos puntos.
    
    Args:
        x1, y1, z1: Coordenadas en t
        x2, y2, z2: Coordenadas en t+dt
        dt: Intervalo de tiempo (segundos)
    
    Returns:
        float: Rapidez en m/s
    """
    if dt == 0:
        return 0.0
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx**2 + dy**2 + dz**2) / dt

def velocidad_vectorial(x1, y1, z1, x2, y2, z2, dt):
    """
    Calcula el vector velocidad entre dos puntos.
    
    Args:
        x1, y1, z1: Coordenadas en t
        x2, y2, z2: Coordenadas en t+dt
        dt: Intervalo de tiempo (segundos)
    
    Returns:
        tuple: (vx, vy, vz) en m/s
    """
    if dt == 0:
        return 0.0, 0.0, 0.0
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    vz = (z2 - z1) / dt
    return vx, vy, vz

def error_velocidad(x, y, z, dx, dy, dz, dt, vx, vy, vz):
    """
    Propaga errores de posición a la velocidad.
    
    Usa la regla de la cadena para estimar el error en velocidad
    a partir de errores en posición.
    
    Args:
        x, y, z: Posición actual (no se usa directamente pero se mantiene por consistencia)
        dx, dy, dz: Errores en posición (metros)
        dt: Intervalo de tiempo (segundos)
        vx, vy, vz: Velocidades actuales (m/s)
    
    Returns:
        float: Error estimado en la velocidad (m/s)
    """
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    if v == 0 or dt == 0:
        return 0.0
    
    # Derivadas parciales de la velocidad respecto a posición
    # dv/dx ≈ (vx/v) * (1/dt)
    dv_dx = (vx / v) / dt
    dv_dy = (vy / v) / dt
    dv_dz = (vz / v) / dt
    
    # Error estimado en velocidad (propagación de errores)
    dv = dv_dx * dx + dv_dy * dy + dv_dz * dz
    
    return dv

def comparacion_errores_velocidad(x1, y1, z1, x2, y2, z2, dt, dx, dy, dz):
    """
    Compara el error exacto vs. la estimación lineal (diferencial) en la
    velocidad, análogo a comparacion_errores() para la distancia.

    El error de medición (dx, dy, dz) se aplica sobre la posición en t+dt,
    tal como se plantea en la sección 5.1 del informe.

    Args:
        x1, y1, z1: Posición real en t
        x2, y2, z2: Posición real en t+dt
        dt: Intervalo de tiempo (segundos)
        dx, dy, dz: Errores de medición en la posición t+dt (metros)

    Returns:
        dict con la velocidad real, la velocidad medida, y la comparación
        entre el error exacto y el error estimado por el diferencial.
    """
    if dt == 0:
        return None

    vx, vy, vz = velocidad_vectorial(x1, y1, z1, x2, y2, z2, dt)
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    # Velocidad calculada a partir de la posición medida (con error) en t+dt
    v_medida = velocidad_instantanea(x1, y1, z1, x2 + dx, y2 + dy, z2 + dz, dt)

    exacto = v_medida - v
    estimado = error_velocidad(x2, y2, z2, dx, dy, dz, dt, vx, vy, vz)

    return {
        'vx': vx, 'vy': vy, 'vz': vz,
        'v_real': v,
        'v_medida': v_medida,
        'error_exacto': exacto,
        'error_estimado': estimado,
        'diferencia_de_error': exacto - estimado,
        'error_relativo_pct': ((exacto - estimado) / exacto * 100) if exacto != 0 else 0,
    }


def analisis_sensibilidad_velocidad(x1, y1, z1, x2, y2, z2, dt, dx, dy, dz):
    """
    Descompone el error estimado en velocidad según la contribución de
    cada componente del error de posición (dx, dy, dz), análogo a
    analisis_sensibilidad_punto() para la distancia.
    """
    vx, vy, vz = velocidad_vectorial(x1, y1, z1, x2, y2, z2, dt)
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    if v == 0 or dt == 0:
        return {'contrib_x': 0.0, 'contrib_y': 0.0, 'contrib_z': 0.0,
                'contrib_x_pct': 0, 'contrib_y_pct': 0, 'contrib_z_pct': 0}

    contrib_x = (vx / v) / dt * dx
    contrib_y = (vy / v) / dt * dy
    contrib_z = (vz / v) / dt * dz
    error_estimado = contrib_x + contrib_y + contrib_z

    if error_estimado == 0:
        return {'contrib_x': contrib_x, 'contrib_y': contrib_y, 'contrib_z': contrib_z,
                'contrib_x_pct': 0, 'contrib_y_pct': 0, 'contrib_z_pct': 0}

    return {
        'contrib_x': contrib_x,
        'contrib_y': contrib_y,
        'contrib_z': contrib_z,
        'contrib_x_pct': abs(contrib_x / error_estimado * 100),
        'contrib_y_pct': abs(contrib_y / error_estimado * 100),
        'contrib_z_pct': abs(contrib_z / error_estimado * 100),
    }


def calcular_velocidad_trayectoria(df, col_tiempo='tiempo'):
    """
    Calcula velocidad y aceleración para una trayectoria.
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z' y col_tiempo
        col_tiempo: Nombre de la columna de tiempo
    
    Returns:
        DataFrame con columnas adicionales de velocidad y aceleración
    """
    df_result = df.copy()
    
    # Calcular diferencias
    df_result['dt'] = df_result[col_tiempo].diff()
    df_result['dx'] = df_result['x'].diff()
    df_result['dy'] = df_result['y'].diff()
    df_result['dz'] = df_result['z'].diff()
    
    # Velocidad instantánea (evitar división por cero)
    df_result['vx'] = 0.0
    df_result['vy'] = 0.0
    df_result['vz'] = 0.0
    
    for i in range(1, len(df_result)):
        dt = df_result.loc[i, 'dt']
        if dt != 0:
            df_result.loc[i, 'vx'] = df_result.loc[i, 'dx'] / dt
            df_result.loc[i, 'vy'] = df_result.loc[i, 'dy'] / dt
            df_result.loc[i, 'vz'] = df_result.loc[i, 'dz'] / dt
    
    df_result['velocidad'] = np.sqrt(
        df_result['vx']**2 + df_result['vy']**2 + df_result['vz']**2
    )
    
    # Aceleración (cambio de velocidad)
    df_result['ax'] = 0.0
    df_result['ay'] = 0.0
    df_result['az'] = 0.0
    
    for i in range(1, len(df_result)):
        dt = df_result.loc[i, 'dt']
        if dt != 0:
            df_result.loc[i, 'ax'] = (df_result.loc[i, 'vx'] - df_result.loc[i-1, 'vx']) / dt
            df_result.loc[i, 'ay'] = (df_result.loc[i, 'vy'] - df_result.loc[i-1, 'vy']) / dt
            df_result.loc[i, 'az'] = (df_result.loc[i, 'vz'] - df_result.loc[i-1, 'vz']) / dt
    
    df_result['aceleracion'] = np.sqrt(
        df_result['ax']**2 + df_result['ay']**2 + df_result['az']**2
    )
    
    # Limpiar valores NaN (primer punto)
    df_result = df_result.fillna(0)
    
    return df_result

def propagacion_error_velocidad_df(df_velocidad, dx, dy, dz):
    """
    Calcula la propagación de errores a velocidad para toda la trayectoria.
    
    Args:
        df_velocidad: DataFrame con columnas 'x','y','z','vx','vy','vz','dt'
        dx, dy, dz: Errores en posición (arrays o listas)
    
    Returns:
        DataFrame con columna 'dv' (error estimado en velocidad)
    """
    df_result = df_velocidad.copy()
    
    errores_dv = []
    
    for i, row in df_result.iterrows():
        # Obtener errores para este punto
        dx_i = dx[i] if i < len(dx) else 0.0
        dy_i = dy[i] if i < len(dy) else 0.0
        dz_i = dz[i] if i < len(dz) else 0.0
        
        # Calcular error en velocidad
        dv = error_velocidad(
            row['x'], row['y'], row['z'],
            dx_i, dy_i, dz_i,
            row['dt'],
            row['vx'], row['vy'], row['vz']
        )
        errores_dv.append(dv)
    
    df_result['dv_estimado'] = errores_dv
    
    return df_result

def error_velocidad_exacto(df_velocidad, dx, dy, dz, col_tiempo='tiempo'):
    """
    Calcula el error exacto en velocidad comparando con y sin errores.
    
    Args:
        df_velocidad: DataFrame con posiciones reales
        dx, dy, dz: Errores en posición
        col_tiempo: Columna de tiempo
    
    Returns:
        DataFrame con columnas de velocidad exacta y con error
    """
    df_result = df_velocidad.copy()
    
    # Crear DataFrame con posiciones con error
    df_con_error = df_result.copy()
    df_con_error['x'] = df_result['x'] + dx
    df_con_error['y'] = df_result['y'] + dy
    df_con_error['z'] = df_result['z'] + dz
    
    # Calcular velocidad con error
    df_con_error = calcular_velocidad_trayectoria(df_con_error, col_tiempo)
    
    # Calcular velocidad real (ya está en df_result)
    df_result = calcular_velocidad_trayectoria(df_result, col_tiempo)
    
    # Error exacto en velocidad
    df_result['velocidad_con_error'] = df_con_error['velocidad']
    df_result['error_velocidad_exacto'] = df_con_error['velocidad'] - df_result['velocidad']
    
    return df_result