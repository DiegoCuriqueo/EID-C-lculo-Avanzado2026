"""
Funciones auxiliares para el proyecto.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

def crear_directorios():
    """
    Crea los directorios necesarios si no existen.
    """
    directorios = [
        'data/resultados',
        'data/raw',
        'data/processed'
    ]
    for dir_path in directorios:
        os.makedirs(dir_path, exist_ok=True)

def guardar_resultados(df, nombre_archivo, subcarpeta='resultados'):
    """
    Guarda un DataFrame en CSV dentro de la carpeta data/.
    
    Args:
        df: DataFrame a guardar
        nombre_archivo: Nombre del archivo (sin extensión)
        subcarpeta: Subcarpeta dentro de data/
    
    Returns:
        str: Ruta del archivo guardado
    """
    crear_directorios()
    ruta = f'data/{subcarpeta}/{nombre_archivo}.csv'
    df.to_csv(ruta, index=False)
    return ruta

def generar_reporte_texto(df_resultados, stats, sensibilidad):
    """
    Genera un reporte en texto con los resultados principales.
    
    Args:
        df_resultados: DataFrame con resultados
        stats: DataFrame con estadísticas
        sensibilidad: DataFrame con análisis de sensibilidad
    
    Returns:
        str: Reporte formateado
    """
    reporte = []
    reporte.append("=" * 70)
    reporte.append("REPORTE DE ANÁLISIS DE PROPAGACIÓN DE ERRORES")
    reporte.append("=" * 70)
    reporte.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"Puntos analizados: {len(df_resultados)}")
    
    # Estadísticas de errores
    reporte.append("\n" + "-" * 50)
    reporte.append("ESTADÍSTICAS DE ERRORES")
    reporte.append("-" * 50)
    reporte.append(f"Error exacto medio: {stats['Error exacto medio'].iloc[0]:.6f} m")
    reporte.append(f"Error exacto std: {stats['Error exacto std'].iloc[0]:.6f} m")
    reporte.append(f"Error estimado medio: {stats['Error estimado medio'].iloc[0]:.6f} m")
    reporte.append(f"Error estimado std: {stats['Error estimado std'].iloc[0]:.6f} m")
    reporte.append(f"Error de aproximación medio: {stats['Error de aproximación medio'].iloc[0]:.6f} m")
    reporte.append(f"Error absoluto medio: {stats['Error absoluto medio'].iloc[0]:.6f} m")
    reporte.append(f"Error máximo: {stats['Error máximo'].iloc[0]:.6f} m")
    reporte.append(f"Correlación exacto-estimado: {stats['Correlación exacto-estimado'].iloc[0]:.6f}")
    
    # Análisis de sensibilidad
    reporte.append("\n" + "-" * 50)
    reporte.append("ANÁLISIS DE SENSIBILIDAD")
    reporte.append("-" * 50)
    for _, row in sensibilidad.iterrows():
        reporte.append(f"{row['Variable']}: {row['Contribución Promedio (%)']:.2f}%")
    
    # Conclusiones
    reporte.append("\n" + "-" * 50)
    reporte.append("📌 CONCLUSIONES")
    reporte.append("-" * 50)
    
    correlacion = stats['Correlación exacto-estimado'].iloc[0]
    if correlacion > 0.95:
        reporte.append("Excelente correlación entre error exacto y estimado")
    elif correlacion > 0.80:
        reporte.append("Buena correlación entre error exacto y estimado")
    elif correlacion > 0.50:
        reporte.append("Correlación moderada - Aproximación útil con limitaciones")
    else:
        reporte.append("Baja correlación - La aproximación no es confiable")
    
    error_medio = stats['Error absoluto medio'].iloc[0]
    if error_medio < 0.01:
        reporte.append("Precisión Excelente de la aproximación")
    elif error_medio < 0.05:
        reporte.append("Buena precisión de la aproximación")
    elif error_medio < 0.10:
        reporte.append("Precisión aceptable para la mayoría de aplicaciones")
    else:
        reporte.append("Precisión limitada - Usar con precaución")
    
    reporte.append("\n" + "=" * 70)
    
    return "\n".join(reporte)

def validar_coordenadas(lat, lon):
    """
    Valida que las coordenadas estén en rangos válidos.
    
    Args:
        lat: Latitud en grados decimales
        lon: Longitud en grados decimales
    
    Returns:
        tuple: (bool, str) - (válido, mensaje)
    """
    if not (-90 <= lat <= 90):
        return False, f"Latitud {lat} fuera de rango (-90, 90)"
    if not (-180 <= lon <= 180):
        return False, f"Longitud {lon} fuera de rango (-180, 180)"
    return True, "Coordenadas válidas"

def formatear_metros(valor):
    """
    Formatea un valor en metros con unidades.
    
    Args:
        valor: Valor en metros
    
    Returns:
        str: Valor formateado con unidades
    """
    if abs(valor) >= 1000:
        return f"{valor/1000:.3f} km"
    elif abs(valor) >= 1:
        return f"{valor:.3f} m"
    else:
        return f"{valor*1000:.1f} mm"

def formatear_tiempo(segundos):
    """
    Formatea un tiempo en segundos a formato legible.
    
    Args:
        segundos: Tiempo en segundos
    
    Returns:
        str: Tiempo formateado
    """
    if segundos < 60:
        return f"{segundos:.1f} s"
    elif segundos < 3600:
        minutos = segundos / 60
        return f"{minutos:.1f} min"
    else:
        horas = segundos / 3600
        return f"{horas:.2f} h"

def guardar_log(mensaje, archivo="data/log.txt"):
    """
    Guarda un mensaje en un archivo de log.
    
    Args:
        mensaje: Mensaje a guardar
        archivo: Ruta del archivo de log
    """
    crear_directorios()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(archivo, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {mensaje}\n")

def generar_csv_ejemplo(ruta="data/ruta_temuco.csv"):
    """
    Genera un archivo CSV de ejemplo con coordenadas en Temuco.
    
    Args:
        ruta: Ruta donde guardar el archivo
    
    Returns:
        str: Ruta del archivo generado
    """
    crear_directorios()
    
    # Coordenadas de una ruta en Temuco
    datos = [
        # lat, lon, altitud, tiempo
        (-38.7359, -72.5904, 50, 0),
        (-38.7365, -72.5895, 52, 1),
        (-38.7370, -72.5885, 48, 2),
        (-38.7372, -72.5870, 55, 3),
        (-38.7368, -72.5855, 60, 4),
        (-38.7360, -72.5840, 58, 5),
        (-38.7350, -72.5830, 62, 6),
        (-38.7340, -72.5825, 65, 7),
        (-38.7330, -72.5820, 70, 8),
        (-38.7320, -72.5815, 68, 9),
        (-38.7310, -72.5810, 72, 10),
        (-38.7300, -72.5815, 75, 11),
        (-38.7290, -72.5820, 78, 12),
        (-38.7280, -72.5830, 80, 13),
        (-38.7270, -72.5840, 82, 14),
        (-38.7260, -72.5850, 85, 15),
        (-38.7270, -72.5860, 88, 16),
        (-38.7280, -72.5870, 90, 17),
        (-38.7290, -72.5880, 88, 18),
        (-38.7300, -72.5890, 85, 19),
        (-38.7310, -72.5900, 82, 20),
        (-38.7320, -72.5905, 78, 21),
        (-38.7330, -72.5900, 75, 22),
        (-38.7340, -72.5900, 72, 23),
        (-38.7350, -72.5902, 70, 24),
        (-38.7359, -72.5904, 68, 25)
    ]
    
    df = pd.DataFrame(datos, columns=['lat', 'lon', 'altitud', 'tiempo'])
    df.to_csv(ruta, index=False)
    return ruta

def cargar_configuracion(ruta="config.json"):
    """
    Carga configuración desde un archivo JSON.
    
    Args:
        ruta: Ruta del archivo JSON
    
    Returns:
        dict: Configuración cargada
    """
    import json
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return {}

def redondear_serie(serie, decimales=4):
    """
    Redondea una serie de pandas a un número de decimales.
    
    Args:
        serie: Serie de pandas
        decimales: Número de decimales
    
    Returns:
        Series: Serie redondeada
    """
    return serie.round(decimales)

def obtener_estadisticas_basicas(df, columnas):
    """
    Obtiene estadísticas básicas de columnas seleccionadas.
    
    Args:
        df: DataFrame
        columnas: Lista de columnas
    
    Returns:
        DataFrame: Estadísticas
    """
    stats = {}
    for col in columnas:
        if col in df.columns:
            stats[col] = {
                'media': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    return pd.DataFrame(stats).T