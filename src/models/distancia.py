"""
Modelo matemático de distancia: D(x,y,z) = sqrt(x² + y² + z²)
"""
import numpy as np

def distancia(x, y, z):
    """Distancia desde el origen (estación base)."""
    return np.sqrt(x**2 + y**2 + z**2)

def distancia_entre_puntos(x1, y1, z1, x2, y2, z2):
    """Distancia entre dos puntos. Útil para extensión temporal."""
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def derivadas_parciales(x, y, z):
    """
    Calcula las tres derivadas parciales.
    Returns: (∂D/∂x, ∂D/∂y, ∂D/∂z)
    """
    D = distancia(x, y, z)
    if D == 0:
        return 0.0, 0.0, 0.0
    return x/D, y/D, z/D

def diferencial_total(x, y, z, dx, dy, dz):
    """
    Error estimado por el diferencial.
    dD = (∂D/∂x)dx + (∂D/∂y)dy + (∂D/∂z)dz
    """
    dDx, dDy, dDz = derivadas_parciales(x, y, z)
    return dDx * dx + dDy * dy + dDz * dz

def error_exacto(x, y, z, dx, dy, dz):
    """Error exacto: D(x+dx, y+dy, z+dz) - D(x, y, z)"""
    return distancia(x + dx, y + dy, z + dz) - distancia(x, y, z)

def comparacion_errores(x, y, z, dx, dy, dz):
    """Compara error exacto vs estimado."""
    exacto = error_exacto(x, y, z, dx, dy, dz)
    estimado = diferencial_total(x, y, z, dx, dy, dz)
    return {
        'error_exacto': exacto,
        'error_estimado': estimado,
        'diferencia': exacto - estimado,
        'error_relativo': ((exacto - estimado)/exacto*100) if exacto != 0 else 0
    }

def error_segundo_orden(x, y, z, dx, dy, dz):
    """
    Calcula el error de segundo orden (curvatura).
    Útil para saber cuándo la aproximación lineal falla.
    """
    D = distancia(x, y, z)
    if D == 0:
        return 0.0
    
    # Términos de segundo orden
    h_xx = (y**2 + z**2) / (D**3)
    h_yy = (x**2 + z**2) / (D**3)
    h_zz = (x**2 + y**2) / (D**3)
    h_xy = -x*y / (D**3)
    h_xz = -x*z / (D**3)
    h_yz = -y*z / (D**3)
    
    # Error de segundo orden
    error_2do = 0.5 * (
        h_xx * dx**2 + h_yy * dy**2 + h_zz * dz**2 +
        2 * h_xy * dx * dy + 2 * h_xz * dx * dz + 2 * h_yz * dy * dz
    )
    
    return error_2do

def precision_aproximacion(x, y, z, dx, dy, dz):
    """Evalúa la precisión de la aproximación lineal."""
    exacto = error_exacto(x, y, z, dx, dy, dz)
    estimado = diferencial_total(x, y, z, dx, dy, dz)
    error_2do = error_segundo_orden(x, y, z, dx, dy, dz)
    residual = exacto - estimado - error_2do
    
    # Clasificación de precisión (considerando error relativo)
    if exacto == 0:
        precision = 'Perfecta'
    else:
        error_relativo = abs(estimado - exacto) / abs(exacto)
        if error_relativo < 0.01:
            precision = 'Excelente'
        elif error_relativo < 0.05:
            precision = 'Buena'
        elif error_relativo < 0.10:
            precision = 'Regular'
        else:
            precision = 'Mala'
    
    return {
        'error_exacto': exacto,
        'error_estimado': estimado,
        'error_2do_orden': error_2do,
        'error_residual': residual,
        'precision': precision
    }

def analisis_sensibilidad_punto(x, y, z, dx, dy, dz):
    """Analiza la contribución de cada error al error total."""
    dDx, dDy, dDz = derivadas_parciales(x, y, z)
    error_total = error_exacto(x, y, z, dx, dy, dz)
    
    contrib_x = dDx * dx
    contrib_y = dDy * dy
    contrib_z = dDz * dz
    
    if error_total == 0:
        return {
            'contrib_x': contrib_x,
            'contrib_y': contrib_y,
            'contrib_z': contrib_z,
            'contrib_x_pct': 0,
            'contrib_y_pct': 0,
            'contrib_z_pct': 0
        }
    
    return {
        'contrib_x': contrib_x,
        'contrib_y': contrib_y,
        'contrib_z': contrib_z,
        'contrib_x_pct': abs(contrib_x / error_total * 100),
        'contrib_y_pct': abs(contrib_y / error_total * 100),
        'contrib_z_pct': abs(contrib_z / error_total * 100)
    }

def calcular_errores_df(df, dx, dy, dz):
    """
    Calcula errores para todo un DataFrame (sin lambda).
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z'
        dx, dy, dz: Arrays o listas con errores para cada punto
    
    Returns:
        DataFrame con columnas adicionales de errores
    """
    # Validar que las columnas existan
    columnas_requeridas = ['x', 'y', 'z']
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"El DataFrame debe tener la columna '{col}'")
    
    # Validar que los errores tengan el tamaño correcto
    n = len(df)
    if len(dx) < n or len(dy) < n or len(dz) < n:
        raise ValueError(f"Los errores deben tener al menos {n} elementos")
    
    df_result = df.copy()
    df_result['dx'] = dx[:n]
    df_result['dy'] = dy[:n]
    df_result['dz'] = dz[:n]
    
    # ===== Distancias =====
    distancias_reales = []
    distancias_medidas = []
    
    for i, row in df_result.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        dx_i, dy_i, dz_i = row['dx'], row['dy'], row['dz']
        
        distancias_reales.append(distancia(x, y, z))
        distancias_medidas.append(distancia(x + dx_i, y + dy_i, z + dz_i))
    
    df_result['distancia_real'] = distancias_reales
    df_result['distancia_medida'] = distancias_medidas
    
    # ===== Errores =====
    df_result['error_exacto'] = df_result['distancia_medida'] - df_result['distancia_real']
    
    errores_estimados = []
    for i, row in df_result.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        dx_i, dy_i, dz_i = row['dx'], row['dy'], row['dz']
        errores_estimados.append(diferencial_total(x, y, z, dx_i, dy_i, dz_i))
    
    df_result['error_estimado'] = errores_estimados
    df_result['error_aproximacion'] = df_result['error_exacto'] - df_result['error_estimado']
    
    # ===== Derivadas parciales =====
    dDx_list = []
    dDy_list = []
    dDz_list = []
    
    for i, row in df_result.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        dDx, dDy, dDz = derivadas_parciales(x, y, z)
        dDx_list.append(dDx)
        dDy_list.append(dDy)
        dDz_list.append(dDz)
    
    df_result['dDx'] = dDx_list
    df_result['dDy'] = dDy_list
    df_result['dDz'] = dDz_list
    
    # ===== Contribuciones =====
    df_result['contrib_x'] = df_result['dDx'] * df_result['dx']
    df_result['contrib_y'] = df_result['dDy'] * df_result['dy']
    df_result['contrib_z'] = df_result['dDz'] * df_result['dz']
    
    # ===== Error de segundo orden =====
    errores_2do = []
    for i, row in df_result.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        dx_i, dy_i, dz_i = row['dx'], row['dy'], row['dz']
        errores_2do.append(error_segundo_orden(x, y, z, dx_i, dy_i, dz_i))
    
    df_result['error_2do_orden'] = errores_2do
    
    return df_result