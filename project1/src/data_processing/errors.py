"""
Módulo para la generación y manejo de errores en coordenadas.
"""
import numpy as np
import pandas as pd

def generar_errores(n, std_xy=0.5, std_z=0.2, correlacion=0):
    """
    Genera errores aleatorios para simular GPS.
    
    Args:
        n: Número de puntos
        std_xy: Desviación estándar en x e y (metros)
        std_z: Desviación estándar en z (metros)
        correlacion: Correlación entre errores (0 = independientes)
    
    Returns:
        tuple: (dx, dy, dz) - Arrays con errores
    """
    if correlacion == 0:
        # Errores independientes
        dx = np.random.normal(0, std_xy, n)
        dy = np.random.normal(0, std_xy, n)
        dz = np.random.normal(0, std_z, n)
    else:
        # Errores correlacionados
        cov = np.array([
            [std_xy**2, correlacion * std_xy * std_xy, 0],
            [correlacion * std_xy * std_xy, std_xy**2, 0],
            [0, 0, std_z**2]
        ])
        errores = np.random.multivariate_normal([0, 0, 0], cov, n)
        dx = errores[:, 0]
        dy = errores[:, 1]
        dz = errores[:, 2]
    
    return dx, dy, dz

def generar_errores_personalizados(n, errores_x, errores_y, errores_z):
    """
    Genera errores con distribuciones personalizadas.
    
    Args:
        n: Número de puntos
        errores_x: Lista o valor fijo para errores en x
        errores_y: Lista o valor fijo para errores en y
        errores_z: Lista o valor fijo para errores en z
    
    Returns:
        tuple: (dx, dy, dz) - Arrays con errores
    """
    # Procesar errores en x
    if isinstance(errores_x, (list, np.ndarray)):
        if len(errores_x) < n:
            dx = np.tile(errores_x, int(np.ceil(n / len(errores_x))))[:n]
        else:
            dx = errores_x[:n]
    else:
        dx = np.full(n, errores_x)
    
    # Procesar errores en y
    if isinstance(errores_y, (list, np.ndarray)):
        if len(errores_y) < n:
            dy = np.tile(errores_y, int(np.ceil(n / len(errores_y))))[:n]
        else:
            dy = errores_y[:n]
    else:
        dy = np.full(n, errores_y)
    
    # Procesar errores en z
    if isinstance(errores_z, (list, np.ndarray)):
        if len(errores_z) < n:
            dz = np.tile(errores_z, int(np.ceil(n / len(errores_z))))[:n]
        else:
            dz = errores_z[:n]
    else:
        dz = np.full(n, errores_z)
    
    return dx, dy, dz

def generar_errores_gps_real(n, calidad='estandar'):
    """
    Genera errores simulando GPS real con diferentes calidades.
    
    Args:
        n: Número de puntos
        calidad: 'estandar', 'preciso', 'impreciso', 'glonass'
    
    Returns:
        tuple: (dx, dy, dz)
    """
    configuraciones = {
        'estandar': {'std_xy': 0.5, 'std_z': 0.2, 'bias': 0.1},
        'preciso': {'std_xy': 0.1, 'std_z': 0.05, 'bias': 0.02},
        'impreciso': {'std_xy': 2.0, 'std_z': 0.5, 'bias': 0.5},
        'glonass': {'std_xy': 0.3, 'std_z': 0.1, 'bias': 0.05}
    }
    
    config = configuraciones.get(calidad, configuraciones['estandar'])
    
    # Generar errores con sesgo
    dx = np.random.normal(config['bias'], config['std_xy'], n)
    dy = np.random.normal(config['bias'], config['std_xy'], n)
    dz = np.random.normal(config['bias'] / 2, config['std_z'], n)
    
    return dx, dy, dz

def agregar_errores_a_dataframe(df, dx, dy, dz):
    """
    Agrega errores a las coordenadas de un DataFrame.
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z'
        dx, dy, dz: Arrays con errores
    
    Returns:
        DataFrame con coordenadas con error
    """
    df_result = df.copy()
    
    # Agregar errores
    df_result['dx'] = dx[:len(df)]
    df_result['dy'] = dy[:len(df)]
    df_result['dz'] = dz[:len(df)]
    
    # Coordenadas con error
    df_result['x_med'] = df_result['x'] + df_result['dx']
    df_result['y_med'] = df_result['y'] + df_result['dy']
    df_result['z_med'] = df_result['z'] + df_result['dz']
    
    return df_result

def calcular_estadisticas_errores_generados(dx, dy, dz):
    """
    Calcula estadísticas de los errores generados.
    
    Args:
        dx, dy, dz: Arrays con errores
    
    Returns:
        dict: Estadísticas
    """
    return {
        'dx_mean': np.mean(dx),
        'dx_std': np.std(dx),
        'dx_min': np.min(dx),
        'dx_max': np.max(dx),
        'dy_mean': np.mean(dy),
        'dy_std': np.std(dy),
        'dy_min': np.min(dy),
        'dy_max': np.max(dy),
        'dz_mean': np.mean(dz),
        'dz_std': np.std(dz),
        'dz_min': np.min(dz),
        'dz_max': np.max(dz),
        'distancia_error_promedio': np.mean(np.sqrt(dx**2 + dy**2 + dz**2))
    }

def generar_errores_con_deriva(n, drift_xy=0.01, drift_z=0.005, std_xy=0.5, std_z=0.2):
    """
    Genera errores con deriva temporal (error acumulativo).
    Útil para simular errores que crecen con el tiempo.
    
    Args:
        n: Número de puntos
        drift_xy: Deriva por paso en x e y (m/paso)
        drift_z: Deriva por paso en z (m/paso)
        std_xy: Desviación estándar base en x e y
        std_z: Desviación estándar base en z
    
    Returns:
        tuple: (dx, dy, dz)
    """
    dx = np.zeros(n)
    dy = np.zeros(n)
    dz = np.zeros(n)
    
    for i in range(n):
        # Error acumulativo con deriva
        dx[i] = dx[i-1] + np.random.normal(drift_xy, std_xy) if i > 0 else np.random.normal(0, std_xy)
        dy[i] = dy[i-1] + np.random.normal(drift_xy, std_xy) if i > 0 else np.random.normal(0, std_xy)
        dz[i] = dz[i-1] + np.random.normal(drift_z, std_z) if i > 0 else np.random.normal(0, std_z)
    
    return dx, dy, dz