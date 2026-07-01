"""Módulo para el análisis de sensibilidad de la función distancia.
"""
import pandas as pd
import numpy as np

def analisis_sensibilidad(df_resultados):
    """
    Analiza la sensibilidad de cada variable (x, y, z).
    
    Args:
        df_resultados: DataFrame con columnas 'contrib_x', 'contrib_y', 'contrib_z', 'error_exacto'
    
    Returns:
        tuple: (DataFrame con sensibilidad, DataFrame con contribuciones porcentuales)
    """
    df = df_resultados.copy()
    
    # Contribución porcentual al error total
    df['contrib_x_pct'] = df.apply(
        lambda row: abs(row['contrib_x'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
        axis=1
    )
    df['contrib_y_pct'] = df.apply(
        lambda row: abs(row['contrib_y'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
        axis=1
    )
    df['contrib_z_pct'] = df.apply(
        lambda row: abs(row['contrib_z'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
        axis=1
    )
    
    # Reemplazar valores infinitos o NaN con 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Estadísticas resumidas de sensibilidad
    sensibilidad = pd.DataFrame({
        'Variable': ['x', 'y', 'z'],
        'Contribución Promedio (%)': [
            df['contrib_x_pct'].mean(),
            df['contrib_y_pct'].mean(),
            df['contrib_z_pct'].mean()
        ],
        'Contribución Máxima (%)': [
            df['contrib_x_pct'].max(),
            df['contrib_y_pct'].max(),
            df['contrib_z_pct'].max()
        ],
        'Contribución Mínima (%)': [
            df['contrib_x_pct'].min(),
            df['contrib_y_pct'].min(),
            df['contrib_z_pct'].min()
        ],
        'Contribución Std (%)': [
            df['contrib_x_pct'].std(),
            df['contrib_y_pct'].std(),
            df['contrib_z_pct'].std()
        ]
    })
    
    return sensibilidad, df

def analisis_por_distancia(df_resultados, n_bins=5):
    """
    Analiza cómo cambia la precisión con la distancia.
    
    Args:
        df_resultados: DataFrame con columnas 'distancia_real', 'error_aproximacion'
        n_bins: Número de rangos de distancia
    
    Returns:
        DataFrame con análisis por rango de distancia
    """
    df = df_resultados.copy()
    
    # Crear rangos de distancia
    df['rango_distancia'] = pd.cut(
        df['distancia_real'],
        bins=n_bins,
        labels=[f'Rango {i+1}' for i in range(n_bins)]
    )
    
    # Análisis por rango
    precision_por_rango = df.groupby('rango_distancia', observed=False).agg({
        'error_aproximacion': ['mean', 'std', 'count'],
        'error_exacto': ['mean', 'std'],
        'error_estimado': ['mean', 'std'],
        'distancia_real': ['mean', 'min', 'max'],
        'error_2do_orden': ['mean']
    }).round(4)
    
    # Agregar error relativo promedio por rango
    df['error_relativo_rango'] = df.apply(
        lambda row: abs(row['error_aproximacion'] / row['error_exacto']) if row['error_exacto'] != 0 else 0,
        axis=1
    )
    
    error_relativo_por_rango = df.groupby('rango_distancia', observed=False)['error_relativo_rango'].mean()
    precision_por_rango[('error_relativo', 'mean')] = error_relativo_por_rango.round(4)
    
    return precision_por_rango

def comparar_sensibilidad_escenarios(df_resultados, escenarios):
    """
    Compara la sensibilidad entre diferentes escenarios de error.
    
    Args:
        df_resultados: DataFrame base
        escenarios: Lista de diccionarios con {'nombre': str, 'dx': array, 'dy': array, 'dz': array}
    
    Returns:
        DataFrame comparativo de sensibilidad
    """
    comparacion = []
    
    for escenario in escenarios:
        # Calcular contribuciones para este escenario
        df_temp = df_resultados.copy()
        df_temp['dx'] = escenario['dx']
        df_temp['dy'] = escenario['dy']
        df_temp['dz'] = escenario['dz']
        
        # Contribuciones (usando derivadas ya calculadas)
        df_temp['contrib_x'] = df_temp['dDx'] * df_temp['dx']
        df_temp['contrib_y'] = df_temp['dDy'] * df_temp['dy']
        df_temp['contrib_z'] = df_temp['dDz'] * df_temp['dz']
        
        # Error exacto
        df_temp['error_exacto'] = df_temp.apply(
            lambda row: np.sqrt((row['x']+row['dx'])**2 + (row['y']+row['dy'])**2 + (row['z']+row['dz'])**2) - row['distancia_real'],
            axis=1
        )
        
        # Contribuciones porcentuales
        df_temp['contrib_x_pct'] = df_temp.apply(
            lambda row: abs(row['contrib_x'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
            axis=1
        )
        df_temp['contrib_y_pct'] = df_temp.apply(
            lambda row: abs(row['contrib_y'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
            axis=1
        )
        df_temp['contrib_z_pct'] = df_temp.apply(
            lambda row: abs(row['contrib_z'] / row['error_exacto'] * 100) if row['error_exacto'] != 0 else 0,
            axis=1
        )
        
        comparacion.append({
            'Escenario': escenario['nombre'],
            'Contribución x (%)': df_temp['contrib_x_pct'].mean(),
            'Contribución y (%)': df_temp['contrib_y_pct'].mean(),
            'Contribución z (%)': df_temp['contrib_z_pct'].mean(),
            'Error exacto medio': df_temp['error_exacto'].mean(),
            'Error estimado medio': (df_temp['contrib_x'] + df_temp['contrib_y'] + df_temp['contrib_z']).mean()
        })
    
    return pd.DataFrame(comparacion)

def analisis_sensibilidad_por_variable(df_resultados):
    """
    Analiza la sensibilidad de cada variable por separado.
    
    Args:
        df_resultados: DataFrame con columnas 'x', 'y', 'z', 'dDx', 'dDy', 'dDz'
    
    Returns:
        dict: Sensibilidad para cada variable
    """
    df = df_resultados.copy()
    
    sensibilidad = {}
    
    # Sensibilidad de x
    sensibilidad['x'] = {
        'promedio': df['dDx'].mean(),
        'max': df['dDx'].max(),
        'min': df['dDx'].min(),
        'std': df['dDx'].std(),
        'interpretacion': 'Alta' if df['dDx'].abs().mean() > 0.5 else 'Baja'
    }
    
    # Sensibilidad de y
    sensibilidad['y'] = {
        'promedio': df['dDy'].mean(),
        'max': df['dDy'].max(),
        'min': df['dDy'].min(),
        'std': df['dDy'].std(),
        'interpretacion': 'Alta' if df['dDy'].abs().mean() > 0.5 else 'Baja'
    }
    
    # Sensibilidad de z
    sensibilidad['z'] = {
        'promedio': df['dDz'].mean(),
        'max': df['dDz'].max(),
        'min': df['dDz'].min(),
        'std': df['dDz'].std(),
        'interpretacion': 'Alta' if df['dDz'].abs().mean() > 0.5 else 'Baja'
    }
    
    return sensibilidad