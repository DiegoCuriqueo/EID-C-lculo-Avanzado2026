"""
Módulo para comparar errores exactos vs estimados.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def comparar_errores(df_resultados):
    """
    Compara errores exactos y estimados.
    
    Args:
        df_resultados: DataFrame con columnas 'error_exacto' y 'error_estimado'
    
    Returns:
        dict: Métricas de comparación
    """
    # Eliminar valores NaN si existen
    df_clean = df_resultados.dropna(subset=['error_exacto', 'error_estimado'])
    
    if len(df_clean) < 2:
        return {
            'error_exacto_mean': 0,
            'error_exacto_std': 0,
            'error_estimado_mean': 0,
            'error_estimado_std': 0,
            'error_aprox_mean': 0,
            'error_aprox_std': 0,
            'error_aprox_abs_mean': 0,
            'error_aprox_abs_max': 0,
            'error_aprox_abs_min': 0,
            'correlacion': 0,
            'p_value': 1
        }
    
    correlacion, p_valor = pearsonr(df_clean['error_exacto'], df_clean['error_estimado'])
    
    comparacion = {
        'error_exacto_mean': df_clean['error_exacto'].mean(),
        'error_exacto_std': df_clean['error_exacto'].std(),
        'error_estimado_mean': df_clean['error_estimado'].mean(),
        'error_estimado_std': df_clean['error_estimado'].std(),
        'error_aprox_mean': df_clean['error_aproximacion'].mean(),
        'error_aprox_std': df_clean['error_aproximacion'].std(),
        'error_aprox_abs_mean': df_clean['error_aproximacion'].abs().mean(),
        'error_aprox_abs_max': df_clean['error_aproximacion'].abs().max(),
        'error_aprox_abs_min': df_clean['error_aproximacion'].abs().min(),
        'correlacion': correlacion,
        'p_value': p_valor
    }
    
    return comparacion

def comparacion_por_distancia(df_resultados, n_bins=5):
    """
    Compara errores agrupados por rangos de distancia.
    
    Args:
        df_resultados: DataFrame con columnas 'distancia_real', 'error_exacto', 'error_estimado', 'error_aproximacion'
        n_bins: Número de rangos de distancia
    
    Returns:
        DataFrame con comparación por rangos
    """
    df = df_resultados.copy()
    
    # Crear bins de distancia
    df['rango_distancia'] = pd.cut(
        df['distancia_real'],
        bins=n_bins,
        labels=[f'Rango {i+1}' for i in range(n_bins)]
    )
    
    comparacion_rangos = df.groupby('rango_distancia', observed=False).agg({
        'error_exacto': ['mean', 'std'],
        'error_estimado': ['mean', 'std'],
        'error_aproximacion': ['mean', 'std', 'count'],
        'distancia_real': ['mean', 'min', 'max']
    }).round(4)
    
    return comparacion_rangos

def metricas_precision(df_resultados):
    """
    Calcula métricas de precisión de la aproximación.
    
    Args:
        df_resultados: DataFrame con columnas 'error_exacto' y 'error_aproximacion'
    
    Returns:
        dict: Métricas de precisión
    """
    df = df_resultados.copy()
    
    # Eliminar valores NaN
    df = df.dropna(subset=['error_exacto', 'error_aproximacion'])
    
    if len(df) == 0:
        return {
            'MAE': 0,
            'RMSE': 0,
            'MAPE': 0,
            'R2': 0,
            'MSE': 0
        }
    
    # Error relativo
    df['error_relativo'] = df.apply(
        lambda row: (row['error_aproximacion'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
        axis=1
    )
    
    # Reemplazar infinitos y NaN
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calcular métricas
    mae = df['error_aproximacion'].abs().mean()
    mse = (df['error_aproximacion']**2).mean()
    rmse = np.sqrt(mse)
    mape = df['error_relativo'].abs().mean()
    
    # R²
    ss_res = (df['error_aproximacion']**2).sum()
    ss_tot = ((df['error_exacto'] - df['error_exacto'].mean())**2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def tabla_comparativa(df_resultados, df_velocidad=None):
    """
    Genera una tabla comparativa completa.
    
    Args:
        df_resultados: DataFrame con resultados de propagación
        df_velocidad: DataFrame con resultados de velocidad (opcional)
    
    Returns:
        DataFrame con tabla comparativa
    """
    # Métricas básicas
    comparacion = comparar_errores(df_resultados)
    metricas = metricas_precision(df_resultados)
    
    # Construir tabla
    tabla = pd.DataFrame({
        'Métrica': [
            'Error exacto medio (m)',
            'Error exacto std (m)',
            'Error estimado medio (m)',
            'Error estimado std (m)',
            'Error de aprox. medio (m)',
            'Error de aprox. std (m)',
            'Error absoluto medio (m)',
            'Error máximo (m)',
            'Error mínimo (m)',
            'Correlación exacto-estimado',
            'Valor p',
            'MAE (m)',
            'RMSE (m)',
            'MSE (m²)',
            'R²',
            'MAPE (%)'
        ],
        'Valor': [
            comparacion['error_exacto_mean'],
            comparacion['error_exacto_std'],
            comparacion['error_estimado_mean'],
            comparacion['error_estimado_std'],
            comparacion['error_aprox_mean'],
            comparacion['error_aprox_std'],
            comparacion['error_aprox_abs_mean'],
            comparacion['error_aprox_abs_max'],
            comparacion['error_aprox_abs_min'],
            comparacion['correlacion'],
            comparacion['p_value'],
            metricas['MAE'],
            metricas['RMSE'],
            metricas['MSE'],
            metricas['R2'],
            metricas['MAPE']
        ]
    })
    
    # Si hay datos de velocidad, agregar
    if df_velocidad is not None:
        tabla_vel = pd.DataFrame({
            'Métrica': [
                'Error velocidad medio (m/s)',
                'Error velocidad std (m/s)',
                'Error velocidad max (m/s)'
            ],
            'Valor': [
                df_velocidad['dv_estimado'].abs().mean(),
                df_velocidad['dv_estimado'].std(),
                df_velocidad['dv_estimado'].abs().max()
            ]
        })
        tabla = pd.concat([tabla, tabla_vel], ignore_index=True)
    
    return tabla

def generar_resumen_comparativo(df_resultados):
    """
    Genera un resumen ejecutivo de la comparación.
    
    Args:
        df_resultados: DataFrame con resultados
    
    Returns:
        dict: Resumen con interpretación
    """
    correlacion = df_resultados[['error_exacto', 'error_estimado']].corr().iloc[0, 1]
    error_medio = df_resultados['error_aproximacion'].abs().mean()
    error_max = df_resultados['error_aproximacion'].abs().max()
    
    # Interpretación de la correlación
    if correlacion > 0.95:
        correlacion_texto = "Excelente"
    elif correlacion > 0.80:
        correlacion_texto = "Buena"
    elif correlacion > 0.50:
        correlacion_texto = "Moderada"
    else:
        correlacion_texto = "Baja"
    
    # Interpretación de la precisión
    if error_medio < 0.01:
        precision_texto = "Excelente"
    elif error_medio < 0.05:
        precision_texto = "Buena"
    elif error_medio < 0.10:
        precision_texto = "Aceptable"
    else:
        precision_texto = "Limitada"
    
    return {
        'correlacion': correlacion,
        'correlacion_texto': correlacion_texto,
        'error_medio': error_medio,
        'error_max': error_max,
        'precision_texto': precision_texto,
        'conclusion': f"La aproximación tiene una correlación {correlacion_texto.lower()} ({correlacion:.3f}) "
                     f"con un error medio de {error_medio:.4f}m, lo que indica una precisión {precision_texto.lower()}."
    }