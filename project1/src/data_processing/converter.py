"""
Módulo para la conversión de coordenadas geográficas a UTM.
"""
import pandas as pd
import utm

def convertir_a_utm(lat, lon):
    """
    Convierte coordenadas geográficas a UTM.
    
    Args:
        lat: Latitud en grados decimales
        lon: Longitud en grados decimales
    
    Returns:
        tuple: (x, y, zona, letra) - Coordenadas UTM
    """
    try:
        x, y, zona, letra = utm.from_latlon(lat, lon)
        return x, y, zona, letra
    except Exception as e:
        raise ValueError(f"Error en conversión UTM para ({lat}, {lon}): {e}")

def convertir_a_latlon(x, y, zona=18, letra='H'):
    """
    Convierte coordenadas UTM a geográficas.
    
    Args:
        x, y: Coordenadas UTM (metros)
        zona: Zona UTM (por defecto 18 para Temuco)
        letra: Letra de la zona UTM (por defecto 'H' para Chile)
    
    Returns:
        tuple: (lat, lon) - Coordenadas geográficas
    """
    try:
        lat, lon = utm.to_latlon(x, y, zona, letra)
        return lat, lon
    except Exception as e:
        raise ValueError(f"Error en conversión a lat/lon para ({x}, {y}): {e}")

def convertir_coordenadas_relativas(df, base_lat=None, base_lon=None):
    """
    Convierte coordenadas geográficas a sistema cartesiano relativo.
    
    Args:
        df: DataFrame con columnas 'lat' y 'lon'
        base_lat: Latitud de la estación base (opcional)
        base_lon: Longitud de la estación base (opcional)
    
    Returns:
        tuple: (df_convertido, base_lat, base_lon)
    """
    # Si no se especifica base, usar el primer punto
    if base_lat is None or base_lon is None:
        base_lat = df.iloc[0]['lat']
        base_lon = df.iloc[0]['lon']
    
    # Convertir base a UTM
    base_x, base_y, zona, letra = convertir_a_utm(base_lat, base_lon)
    base_z = 0
    
    # Guardar zona UTM para conversiones posteriores
    zona_utm = zona
    letra_utm = letra
    
    # Convertir todas las coordenadas
    x_list = []
    y_list = []
    z_list = []
    lat_list = []
    lon_list = []
    
    for i, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        
        # Convertir a UTM
        x, y, _, _ = convertir_a_utm(lat, lon)
        
        # Coordenadas relativas a la base
        x_rel = x - base_x
        y_rel = y - base_y
        
        # Altitud (si existe, sino 0)
        if 'altitud' in row.index:
            z_rel = row['altitud'] - base_z
        else:
            z_rel = 0.0
        
        x_list.append(x_rel)
        y_list.append(y_rel)
        z_list.append(z_rel)
        lat_list.append(lat)
        lon_list.append(lon)
    
    # Crear nuevo DataFrame
    df_convertido = pd.DataFrame({
        'x': x_list,
        'y': y_list,
        'z': z_list,
        'lat': lat_list,
        'lon': lon_list
    })
    
    # Agregar columna de tiempo si existe
    if 'tiempo' in df.columns:
        df_convertido['tiempo'] = df['tiempo'].values
    
    # Guardar información de la base
    df_convertido.attrs['base_x'] = base_x
    df_convertido.attrs['base_y'] = base_y
    df_convertido.attrs['base_lat'] = base_lat
    df_convertido.attrs['base_lon'] = base_lon
    df_convertido.attrs['zona_utm'] = zona_utm
    df_convertido.attrs['letra_utm'] = letra_utm
    
    return df_convertido, base_lat, base_lon

def convertir_a_geograficas(df, zona=18, letra='H'):
    """
    Convierte coordenadas relativas de vuelta a geográficas.
    
    Args:
        df: DataFrame con columnas 'x', 'y' (relativas)
        zona: Zona UTM
        letra: Letra de la zona UTM
    
    Returns:
        DataFrame con columnas 'lat', 'lon'
    """
    # Obtener base desde atributos del DataFrame
    base_x = df.attrs.get('base_x', 0)
    base_y = df.attrs.get('base_y', 0)
    
    lat_list = []
    lon_list = []
    
    for i, row in df.iterrows():
        x_abs = row['x'] + base_x
        y_abs = row['y'] + base_y
        
        lat, lon = convertir_a_latlon(x_abs, y_abs, zona, letra)
        lat_list.append(lat)
        lon_list.append(lon)
    
    df_result = df.copy()
    df_result['lat'] = lat_list
    df_result['lon'] = lon_list
    
    return df_result

def obtener_zona_utm(lat, lon):
    """
    Obtiene la zona UTM a partir de coordenadas geográficas.
    
    Args:
        lat: Latitud en grados decimales
        lon: Longitud en grados decimales
    
    Returns:
        tuple: (zona, letra)
    """
    _, _, zona, letra = utm.from_latlon(lat, lon)
    return zona, letra