"""
Módulo para la carga de archivos CSV.
"""
import pandas as pd

def cargar_csv(ruta):
    """
    Carga un archivo CSV con coordenadas.
    
    Args:
        ruta: Ruta del archivo CSV
    
    Returns:
        DataFrame con los datos cargados

    """
    try:
        df = pd.read_csv(ruta)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    except Exception as e:
        raise Exception(f"Error al leer el archivo: {e}")
    
    # Verificar columnas necesarias
    columnas_requeridas = ['lat', 'lon']
    columnas_faltantes = []
    
    for col in columnas_requeridas:
        if col not in df.columns:
            columnas_faltantes.append(col)
    
    if columnas_faltantes:
        raise ValueError(
            f"El archivo debe tener las columnas: {', '.join(columnas_requeridas)}\n"
            f"Columnas faltantes: {', '.join(columnas_faltantes)}"
        )
    
    # Verificar que no haya datos nulos en columnas requeridas
    for col in columnas_requeridas:
        if df[col].isnull().any():
            raise ValueError(f"La columna '{col}' contiene valores nulos")
    
    return df

def cargar_csv_con_altitud(ruta):
    """
    Carga un archivo CSV con coordenadas y altitud.
    
    Args:
        ruta: Ruta del archivo CSV
    
    Returns:
        DataFrame con los datos cargados
    """
    df = cargar_csv(ruta)
    
    # Si no tiene columna 'altitud', agregarla con valor 0
    if 'altitud' not in df.columns:
        df['altitud'] = 0.0
    
    # Si no tiene columna 'tiempo', agregar índice como tiempo
    if 'tiempo' not in df.columns:
        df['tiempo'] = range(len(df))
    
    return df

def validar_coordenadas(df):
    """
    Valida que las coordenadas estén en rangos válidos.
    
    Args:
        df: DataFrame con columnas 'lat' y 'lon'
    
    Returns:
        tuple: (bool, list) - (válido, lista de errores)
    """
    errores = []
    
    for i, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        
        if not (-90 <= lat <= 90):
            errores.append(f"Fila {i}: Latitud {lat} fuera de rango (-90, 90)")
        
        if not (-180 <= lon <= 180):
            errores.append(f"Fila {i}: Longitud {lon} fuera de rango (-180, 180)")
    
    if errores:
        return False, errores
    
    return True, []