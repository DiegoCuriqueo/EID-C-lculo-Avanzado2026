"""
Análisis de cómo el error de medición de posición afecta la trayectoria
planificada del dron hacia un destino (extensión del modelo, sección 5.2
del enunciado: incorporar cómo se propaga el error hacia la posición
cuando el dron se desplaza).
"""
import numpy as np


def trayectoria_con_error(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest):
    """
    Simula el efecto del error de medición sobre la ruta que sigue el dron.

    El sistema de navegación del dron NO conoce su posición real: solo
    cuenta con la posición MEDIDA (posición real + error). A partir de esa
    posición medida, calcula la dirección y la distancia necesarias para
    llegar al destino planeado.

    Sin embargo, el dron físicamente parte desde su posición REAL (no la
    medida). Al recorrer esa misma dirección y distancia calculadas,
    termina llegando a un punto distinto del destino: esa diferencia es
    la desviación provocada por el error de posicionamiento.

    Args:
        x, y, z: posición real del dron
        dx, dy, dz: error de medición en cada coordenada
        x_dest, y_dest, z_dest: coordenadas del destino planeado

    Returns:
        dict con las posiciones involucradas (como np.array) y las
        magnitudes relevantes de la trayectoria.
    """
    real = np.array([x, y, z], dtype=float)
    medida = real + np.array([dx, dy, dz], dtype=float)
    destino = np.array([x_dest, y_dest, z_dest], dtype=float)

    # Dirección y distancia que el dron CREE que debe recorrer,
    # calculadas a partir de la posición medida (con error).
    vector_planeado = destino - medida
    distancia_planeada = np.linalg.norm(vector_planeado)

    if distancia_planeada > 1e-9:
        direccion_planeada = vector_planeado / distancia_planeada
    else:
        direccion_planeada = np.zeros(3)

    # El dron ejecuta esa dirección/distancia, pero partiendo realmente
    # desde su posición real -> llega a un punto distinto del destino.
    punto_llegada = real + direccion_planeada * distancia_planeada
    desviacion = np.linalg.norm(punto_llegada - destino)

    # Distancia ideal si no hubiese error de medición (real -> destino directo)
    distancia_ideal = np.linalg.norm(destino - real)

    return {
        "real": real,
        "medida": medida,
        "destino": destino,
        "punto_llegada": punto_llegada,
        "direccion_planeada": direccion_planeada,
        "distancia_planeada": distancia_planeada,
        "distancia_ideal": distancia_ideal,
        "desviacion": desviacion,
    }
