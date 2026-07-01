"""
Constantes globales para el proyecto.
"""

# ==================== UTM ====================
UTM_ZONA = 18
UTM_LETRA = 'H'

# ==================== ERRORES TÍPICOS ====================
ERROR_GPS_ESTANDAR_STD_XY = 0.5
ERROR_GPS_ESTANDAR_STD_Z = 0.2
ERROR_GPS_PRECISO_STD_XY = 0.1
ERROR_GPS_PRECISO_STD_Z = 0.05
ERROR_GPS_IMPRECISO_STD_XY = 2.0
ERROR_GPS_IMPRECISO_STD_Z = 0.5
ERROR_GLONASS_STD_XY = 0.3
ERROR_GLONASS_STD_Z = 0.1

# ==================== COORDENADAS BASE ====================
TEMUCO_LAT = -38.7359
TEMUCO_LON = -72.5904
TEMUCO_ALT = 0

# ==================== VISUALIZACIÓN ====================
COLOR_RUTA_REAL = '#00cc66'
COLOR_RUTA_MEDIDA = '#ff4444'
COLOR_BASE = '#ff0000'
COLOR_GRADIENTE = '#0066ff'
COLOR_ERROR = '#ff8800'

# ==================== STREAMLIT ====================
STREAMLIT_TITLE = "Análisis de Propagación de Errores en Drones"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_STATE = "expanded"

# ==================== ARCHIVOS Y CARPETAS ====================
DEFAULT_CSV = "data/ruta_temuco.csv"
RESULTADOS_DIR = "data/resultados/"
PROCESSED_DIR = "data/processed/"
RAW_DIR = "data/raw/"

# ==================== UMBRALES Y TOLERANCIAS ====================
TOLERANCIA_PERpendicular = 1e-6
TOLERANCIA_UNITARIO = 1e-6
TOLERANCIA_ERROR = 1e-10

# ==================== CLASIFICACIÓN DE PRECISIÓN ====================
PRECISION_UMBRALES = {
    'excelente': 0.01,
    'buena': 0.05,
    'regular': 0.10,
    'mala': float('inf')
}