"""
Módulo para gráficos de errores con Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.constants import COLOR_RUTA_REAL, COLOR_RUTA_MEDIDA, COLOR_ERROR

def graficar_comparacion_errores(df_resultados, guardar=True):
    """
    Genera gráficos de comparación de errores.
    
    Args:
        df_resultados: DataFrame con resultados
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    # Crear carpeta si no existe
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error exacto vs estimado
    ax1 = axes[0, 0]
    ax1.scatter(df_resultados['error_exacto'], df_resultados['error_estimado'], 
                alpha=0.6, s=50, c='blue')
    
    min_val = min(df_resultados['error_exacto'].min(), df_resultados['error_estimado'].min())
    max_val = max(df_resultados['error_exacto'].max(), df_resultados['error_estimado'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfecta correlación')
    ax1.set_xlabel('Error Exacto (m)', fontsize=12)
    ax1.set_ylabel('Error Estimado (m)', fontsize=12)
    ax1.set_title('Comparación: Error Exacto vs Estimado', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Distribución del error de aproximación
    ax2 = axes[0, 1]
    ax2.hist(df_resultados['error_aproximacion'], bins=20, edgecolor='black', 
             alpha=0.7, color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error cero')
    ax2.set_xlabel('Error de Aproximación (m)', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución del Error de Aproximación', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Contribución de cada variable al error
    ax3 = axes[1, 0]
    ax3.plot(df_resultados.index, df_resultados['contrib_x'], 
             'b-', label='Error en x', linewidth=2)
    ax3.plot(df_resultados.index, df_resultados['contrib_y'], 
             'g-', label='Error en y', linewidth=2)
    ax3.plot(df_resultados.index, df_resultados['contrib_z'], 
             'orange', label='Error en z', linewidth=2)
    ax3.set_xlabel('Punto en la trayectoria', fontsize=12)
    ax3.set_ylabel('Contribución al error (m)', fontsize=12)
    ax3.set_title('Contribución de cada coordenada al error', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precisión de la aproximación vs Distancia
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_resultados['distancia_real'], 
                         np.abs(df_resultados['error_aproximacion']),
                         c=df_resultados['distancia_real'], 
                         cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Distancia a la base (m)', fontsize=12)
    ax4.set_ylabel('|Error de Aproximación| (m)', fontsize=12)
    ax4.set_title('Precisión de la aproximación vs Distancia', fontsize=14)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Distancia (m)')
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/graficos_comparacion.png', dpi=300, bbox_inches='tight')
    
    return fig

def graficar_evolucion_temporal(df_resultados, df_original, guardar=True):
    """
    Genera gráfico de evolución temporal de la distancia y errores.
    
    Args:
        df_resultados: DataFrame con resultados
        df_original: DataFrame original con columna 'tiempo'
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Obtener tiempos
    if 'tiempo' in df_original.columns:
        tiempos = df_original['tiempo'].values
    else:
        tiempos = range(len(df_resultados))
    
    # 1. Evolución de la distancia
    ax1 = axes[0]
    ax1.plot(tiempos, df_resultados['distancia_real'], 
             'b-', label='Distancia Real', linewidth=2)
    ax1.plot(tiempos, df_resultados['distancia_medida'], 
             'r-', label='Distancia Medida', linewidth=2, alpha=0.7)
    ax1.fill_between(tiempos, 
                     df_resultados['distancia_medida'],
                     df_resultados['distancia_real'],
                     alpha=0.3, color='red', label='Error de medición')
    ax1.set_xlabel('Tiempo (s)', fontsize=12)
    ax1.set_ylabel('Distancia (m)', fontsize=12)
    ax1.set_title('Evolución temporal de la distancia', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error exacto vs estimado en el tiempo
    ax2 = axes[1]
    ax2.plot(tiempos, df_resultados['error_exacto'], 
             'b-', label='Error Exacto', linewidth=2)
    ax2.plot(tiempos, df_resultados['error_estimado'], 
             'r--', label='Error Estimado (Diferencial)', linewidth=2)
    ax2.fill_between(tiempos, 
                     df_resultados['error_exacto'],
                     df_resultados['error_estimado'],
                     alpha=0.3, color='green', label='Error de aprox.')
    ax2.set_xlabel('Tiempo (s)', fontsize=12)
    ax2.set_ylabel('Error (m)', fontsize=12)
    ax2.set_title('Comparación de errores en el tiempo', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/evolucion_temporal.png', dpi=300, bbox_inches='tight')
    
    return fig

def graficar_errores_por_distancia(df_resultados, guardar=True):
    """
    Genera gráfico de errores agrupados por distancia.
    
    Args:
        df_resultados: DataFrame con resultados
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordenar por distancia
    df_sorted = df_resultados.sort_values('distancia_real')
    
    # Barras de error
    ax.bar(df_sorted.index, df_sorted['error_exacto'], 
           alpha=0.5, label='Error Exacto', color='blue')
    ax.bar(df_sorted.index, df_sorted['error_estimado'], 
           alpha=0.5, label='Error Estimado', color='red')
    
    # Línea de distancia
    ax2 = ax.twinx()
    ax2.plot(df_sorted.index, df_sorted['distancia_real'], 
             'g-', linewidth=2, label='Distancia')
    ax2.set_ylabel('Distancia (m)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax.set_xlabel('Puntos ordenados por distancia', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Errores vs Distancia a la base', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/errores_por_distancia.png', dpi=300, bbox_inches='tight')
    
    return fig

def graficar_sensibilidad(df_sensibilidad, guardar=True):
    """
    Genera gráfico de barras para la sensibilidad.
    
    Args:
        df_sensibilidad: DataFrame con análisis de sensibilidad
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    variables = df_sensibilidad['Variable'].values
    contribuciones = df_sensibilidad['Contribución Promedio (%)'].values
    
    colores = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    barras = ax.bar(variables, contribuciones, color=colores, alpha=0.8)
    
    # Agregar valores encima de las barras
    for barra, valor in zip(barras, contribuciones):
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.5,
                f'{valor:.1f}%', ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('Contribución al error (%)', fontsize=12)
    ax.set_title('Sensibilidad de cada coordenada', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/sensibilidad.png', dpi=300, bbox_inches='tight')
    
    return fig

def graficar_todas_visualizaciones(df_resultados, df_original=None, df_sensibilidad=None):
    """
    Genera todas las visualizaciones a la vez.
    
    Args:
        df_resultados: DataFrame con resultados
        df_original: DataFrame original (para tiempos)
        df_sensibilidad: DataFrame con sensibilidad
    
    Returns:
        dict: Diccionario con todas las figuras
    """
    figuras = {}
    
    # Gráfico de comparación
    figuras['comparacion'] = graficar_comparacion_errores(df_resultados, guardar=True)
    
    # Evolución temporal
    if df_original is not None:
        figuras['temporal'] = graficar_evolucion_temporal(df_resultados, df_original, guardar=True)
    
    # Errores por distancia
    figuras['distancia'] = graficar_errores_por_distancia(df_resultados, guardar=True)
    
    # Sensibilidad
    if df_sensibilidad is not None:
        figuras['sensibilidad'] = graficar_sensibilidad(df_sensibilidad, guardar=True)
    
    return figuras