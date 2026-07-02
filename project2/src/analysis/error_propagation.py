import numpy as np

def trayectoria_con_error(x, y, z, dx, dy, dz, x_dest, y_dest, z_dest):
    """ Simula el impacto del error de posicionamiento sobre la ruta del dron """
    real = np.array([x, y, z], dtype=float)
    medida = real + np.array([dx, dy, dz], dtype=float)
    destino = np.array([x_dest, y_dest, z_dest], dtype=float)

    # Vector y distancia calculados por el dron usando la posición medida
    vector_planeado = destino - medida
    distancia_planeada = np.linalg.norm(vector_planeado)

    if distancia_planeada > 1e-9:
        direccion_planeada = vector_planeado / distancia_planeada
    else:
        direccion_planeada = np.zeros(3)

    # Punto de llegada real al aplicar el plan desde la posición verdadera
    punto_llegada = real + direccion_planeada * distancia_planeada
    desviacion = np.linalg.norm(punto_llegada - destino)

    # Distancia en línea recta sin errores
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