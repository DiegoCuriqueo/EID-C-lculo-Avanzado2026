"""
Módulo de utilidades generales.
"""
from .constants import (
    UTM_ZONA,
    UTM_LETRA,
    ERROR_GPS_ESTANDAR_STD_XY,
    ERROR_GPS_ESTANDAR_STD_Z,
    ERROR_GPS_PRECISO_STD_XY,
    ERROR_GPS_PRECISO_STD_Z,
    ERROR_GPS_IMPRECISO_STD_XY,
    ERROR_GPS_IMPRECISO_STD_Z,
    ERROR_GLONASS_STD_XY,
    ERROR_GLONASS_STD_Z,
    TEMUCO_LAT,
    TEMUCO_LON,
    TEMUCO_ALT,
    COLOR_RUTA_REAL,
    COLOR_RUTA_MEDIDA,
    COLOR_BASE,
    COLOR_GRADIENTE,
    COLOR_ERROR,
    STREAMLIT_TITLE,
    DEFAULT_CSV,
    RESULTADOS_DIR,
    TOLERANCIA_PERpendicular,
    TOLERANCIA_UNITARIO,
    PRECISION_UMBRALES
)

from .helpers import (
    crear_directorios,
    guardar_resultados,
    generar_reporte_texto,
    validar_coordenadas,
    formatear_metros,
    formatear_tiempo,
    guardar_log,
    generar_csv_ejemplo,
    cargar_configuracion,
    redondear_serie,
    obtener_estadisticas_basicas
)