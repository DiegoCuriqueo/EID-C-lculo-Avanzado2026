"""
Módulo para el análisis de propagación de errores.
"""
import pandas as pd
import numpy as np
from src.models.distancia import (
    distancia, 
    diferencial_total, 
    error_exacto, 
    derivadas_parciales,
    error_segundo_orden
)

def analizar_propagacion_errores(df, dx, dy, dz):
    """
    Analiza la propagación de errores para cada punto de la trayectoria.
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z'
        dx, dy, dz: Arrays con errores para cada punto
    
    Returns:
        DataFrame con columnas de análisis completo
    """
    # Validar entradas
    if len(dx) < len(df) or len(dy) < len(df) or len(dz) < len(df):
        raise ValueError("Los arrays de errores deben tener al menos el tamaño del DataFrame")
    
    df_result = df.copy()
    df_result['dx'] = dx[:len(df)]
    df_result['dy'] = dy[:len(df)]
    df_result['dz'] = dz[:len(df)]
    
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

def calcular_estadisticas_errores(df_resultados):
    """
    Calcula estadísticas resumidas de los errores.
    
    Args:
        df_resultados: DataFrame resultante de analizar_propagacion_errores()
    
    Returns:
        DataFrame con estadísticas
    """
    stats = {
        'Error exacto medio': df_resultados['error_exacto'].mean(),
        'Error exacto std': df_resultados['error_exacto'].std(),
        'Error estimado medio': df_resultados['error_estimado'].mean(),
        'Error estimado std': df_resultados['error_estimado'].std(),
        'Error de aproximación medio': df_resultados['error_aproximacion'].mean(),
        'Error de aproximación std': df_resultados['error_aproximacion'].std(),
        'Error absoluto medio': df_resultados['error_aproximacion'].abs().mean(),
        'Error máximo': df_resultados['error_aproximacion'].abs().max(),
        'Error mínimo': df_resultados['error_aproximacion'].abs().min(),
        'Error 2do orden medio': df_resultados['error_2do_orden'].mean(),
        'Correlación exacto-estimado': df_resultados[['error_exacto', 'error_estimado']].corr().iloc[0, 1]
    }
    return pd.DataFrame([stats])

def calcular_errores_por_punto(df_resultados):
    """
    Calcula el error relativo y clasificación por punto.
    
    Args:
        df_resultados: DataFrame resultante de analizar_propagacion_errores()
    
    Returns:
        DataFrame con columnas adicionales
    """
    df = df_resultados.copy()
    
    # Error relativo
    errores_relativos = []
    for i, row in df.iterrows():
        if row['error_exacto'] != 0:
            errores_relativos.append(row['error_aproximacion'] / row['error_exacto'] * 100)
        else:
            errores_relativos.append(0.0)
    
    df['error_relativo'] = errores_relativos
    
    # Clasificación de precisión por punto
    precisiones = []
    for i, row in df.iterrows():
        if row['error_exacto'] == 0:
            precisiones.append('Perfecta')
        else:
            error_rel = abs(row['error_aproximacion']) / abs(row['error_exacto'])
            if error_rel < 0.01:
                precisiones.append('Excelente')
            elif error_rel < 0.05:
                precisiones.append('Buena')
            elif error_rel < 0.10:
                precisiones.append('Regular')
            else:
                precisiones.append('Mala')
    
    df['precision'] = precisiones
    
    return df