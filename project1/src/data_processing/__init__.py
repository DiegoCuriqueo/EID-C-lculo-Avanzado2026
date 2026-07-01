"""
Módulo de procesamiento de datos.
"""
from .loader import (
    cargar_csv,
    cargar_csv_con_altitud,
    validar_coordenadas
)

from .converter import (
    convertir_a_utm,
    convertir_a_latlon,
    convertir_coordenadas_relativas,
    convertir_a_geograficas,
    obtener_zona_utm
)

from .errors import (
    generar_errores,
    generar_errores_personalizados,
    generar_errores_gps_real,
    agregar_errores_a_dataframe,
    calcular_estadisticas_errores_generados,
    generar_errores_con_deriva
)