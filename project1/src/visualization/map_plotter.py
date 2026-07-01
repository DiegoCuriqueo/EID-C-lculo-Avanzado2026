"""
Módulo para visualización de mapas interactivos con Folium.
"""
import folium
import numpy as np
import utm
from src.utils.constants import (
    COLOR_RUTA_REAL,
    COLOR_RUTA_MEDIDA,
    COLOR_BASE,
    COLOR_GRADIENTE
)

def crear_mapa_interactivo(df_resultados, base_lat, base_lon, zona=18, letra='H'):
    """
    Crea un mapa interactivo con la ruta real y medida.
    
    Args:
        df_resultados: DataFrame con resultados del análisis
        base_lat: Latitud de la estación base
        base_lon: Longitud de la estación base
        zona: Zona UTM (por defecto 18 para Temuco)
        letra: Letra UTM (por defecto 'H' para Chile)
    
    Returns:
        folium.Map: Mapa interactivo
    """
    # Mapa centrado en la base
    mapa = folium.Map(
        location=[base_lat, base_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Estación base
    folium.Marker(
        [base_lat, base_lon],
        popup='🗼 Estación Base',
        tooltip='Estación Base',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(mapa)
    
    # Ruta real (coordenadas geográficas originales)
    if 'lat' in df_resultados.columns and 'lon' in df_resultados.columns:
        coordenadas_reales = df_resultados[['lat', 'lon']].values.tolist()
        folium.PolyLine(
            coordenadas_reales,
            color=COLOR_RUTA_REAL,
            weight=3,
            opacity=0.8,
            popup='🟢 Ruta Real (sin errores)',
            tooltip='Ruta Real'
        ).add_to(mapa)
    
    # Ruta medida (convertir x,y a lat/lon)
    if 'x' in df_resultados.columns and 'y' in df_resultados.columns:
        # Obtener base desde atributos o usar los valores proporcionados
        base_x = df_resultados.attrs.get('base_x', 0)
        base_y = df_resultados.attrs.get('base_y', 0)
        
        lat_med_list = []
        lon_med_list = []
        
        for i, row in df_resultados.iterrows():
            x_abs = row['x'] + row.get('dx', 0) + base_x
            y_abs = row['y'] + row.get('dy', 0) + base_y
            
            lat, lon = convertir_a_latlon(x_abs, y_abs, zona, letra)
            lat_med_list.append(lat)
            lon_med_list.append(lon)
        
        coordenadas_medidas = list(zip(lat_med_list, lon_med_list))
        
        folium.PolyLine(
            coordenadas_medidas,
            color=COLOR_RUTA_MEDIDA,
            weight=3,
            opacity=0.8,
            dash_array='5',
            popup='🔴 Ruta Medida (con errores)',
            tooltip='Ruta con errores GPS'
        ).add_to(mapa)
    
    return mapa

def convertir_a_latlon(x, y, zona=18, letra='H'):
    """
    Convierte coordenadas UTM a geográficas.
    
    Args:
        x, y: Coordenadas UTM (metros)
        zona: Zona UTM
        letra: Letra de la zona UTM
    
    Returns:
        tuple: (lat, lon)
    """
    try:
        lat, lon = utm.to_latlon(x, y, zona, letra)
        return lat, lon
    except Exception:
        return 0.0, 0.0

def agregar_puntos_al_mapa(mapa, df_resultados, base_lat, base_lon):
    """
    Agrega marcadores de puntos al mapa.
    
    Args:
        mapa: Objeto folium.Map
        df_resultados: DataFrame con resultados
        base_lat: Latitud de la base
        base_lon: Longitud de la base
    
    Returns:
        folium.Map: Mapa con puntos agregados
    """
    # Puntos reales
    for i, row in df_resultados.iterrows():
        if 'lat' in row and 'lon' in row:
            folium.CircleMarker(
                [row['lat'], row['lon']],
                radius=4,
                color='green',
                fill=True,
                fill_opacity=0.7,
                popup=f"Real: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})"
            ).add_to(mapa)
    
    return mapa

def crear_mapa_con_errores(df_resultados, base_lat, base_lon, zona=18, letra='H'):
    """
    Crea un mapa mostrando los errores como flechas.
    
    Args:
        df_resultados: DataFrame con resultados
        base_lat: Latitud de la base
        base_lon: Longitud de la base
        zona: Zona UTM
        letra: Letra UTM
    
    Returns:
        folium.Map: Mapa con errores
    """
    mapa = crear_mapa_interactivo(df_resultados, base_lat, base_lon, zona, letra)
    
    # Agregar flechas de error para algunos puntos
    for i, row in df_resultados.iterrows():
        if i % 3 == 0:  # Mostrar cada 3 puntos para no saturar
            lat_real = row.get('lat', 0)
            lon_real = row.get('lon', 0)
            
            # Error en distancia
            error = row.get('error_exacto', 0)
            
            if abs(error) > 0.01:  # Solo si el error es significativo
                folium.CircleMarker(
                    [lat_real, lon_real],
                    radius=3,
                    color=COLOR_ERROR,
                    fill=True,
                    fill_opacity=0.5,
                    popup=f"Error: {error:.4f} m"
                ).add_to(mapa)
    
    return mapa