"""
Módulo de análisis para la propagación de errores.
"""
from .error_propagation import (
    analizar_propagacion_errores,
    calcular_estadisticas_errores,
    calcular_errores_por_punto
)

from .sensitivity import (
    analisis_sensibilidad,
    analisis_por_distancia,
    comparar_sensibilidad_escenarios,
    analisis_sensibilidad_por_variable
)

from .comparison import (
    comparar_errores,
    comparacion_por_distancia,
    metricas_precision,
    tabla_comparativa,
    generar_resumen_comparativo
)