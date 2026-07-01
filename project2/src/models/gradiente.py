"""
Módulo para el cálculo y análisis del gradiente de la función distancia.
"""
import numpy as np
from .distancia import distancia

def gradiente(x, y, z):
    """
    Calcula el vector gradiente de la función distancia.
    
    Args:
        x, y, z: Coordenadas del punto
    
    Returns:
        np.array: Vector gradiente [dD/dx, dD/dy, dD/dz]
    
    Nota: El gradiente apunta radialmente desde el origen y es unitario.
    """
    D = distancia(x, y, z)
    if D == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.array([x/D, y/D, z/D])

def magnitud_gradiente(x, y, z):
    """
    Calcula la magnitud del gradiente.
    Siempre es 1 (excepto en el origen).
    
    Args:
        x, y, z: Coordenadas del punto
    
    Returns:
        float: Magnitud del gradiente (0 en el origen, 1 en cualquier otro punto)
    """
    grad = gradiente(x, y, z)
    return np.linalg.norm(grad)

def direccion_maximo_crecimiento(x, y, z):
    """
    Retorna la dirección de máximo crecimiento de la distancia.
    Es equivalente al gradiente normalizado.
    
    Args:
        x, y, z: Coordenadas del punto
    
    Returns:
        np.array: Vector unitario en dirección de máximo crecimiento
    """
    grad = gradiente(x, y, z)
    norm = np.linalg.norm(grad)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return grad / norm

def gradiente_en_puntos(df):
    """
    Calcula el gradiente para cada punto en un DataFrame.
    
    Args:
        df: DataFrame con columnas 'x', 'y', 'z'
    
    Returns:
        DataFrame con columnas adicionales 'grad_x', 'grad_y', 'grad_z'
    """
    # Verificar que las columnas existan
    columnas_requeridas = ['x', 'y', 'z']
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"El DataFrame debe tener la columna '{col}'")
    
    df_result = df.copy()
    
    grad_x_list = []
    grad_y_list = []
    grad_z_list = []
    
    for i, row in df_result.iterrows():
        gx, gy, gz = gradiente(row['x'], row['y'], row['z'])
        grad_x_list.append(gx)
        grad_y_list.append(gy)
        grad_z_list.append(gz)
    
    df_result['grad_x'] = grad_x_list
    df_result['grad_y'] = grad_y_list
    df_result['grad_z'] = grad_z_list
    
    return df_result

def perpendicularidad_gradiente_nivel(x, y, z):
    """
    Verifica que el gradiente es perpendicular a las superficies de nivel.
    
    Para la función distancia, las superficies de nivel son esferas centradas en el origen.
    El gradiente apunta radialmente (perpendicular a la esfera).
    
    Args:
        x, y, z: Coordenadas del punto
    
    Returns:
        dict: {
            'producto_punto': float,
            'es_perpendicular': bool,
            'angulo': float,
            'interpretacion': str
        }
    """
    grad = gradiente(x, y, z)
    radio = np.array([x, y, z])
    
    producto_punto = np.dot(grad, radio)
    
    norm_grad = np.linalg.norm(grad)
    norm_radio = np.linalg.norm(radio)
    
    if norm_grad == 0 or norm_radio == 0:
        angulo = 0
    else:
        cos_angulo = producto_punto / (norm_grad * norm_radio)
        cos_angulo = max(-1, min(1, cos_angulo))
        angulo = np.arccos(cos_angulo) * 180 / np.pi
    
    es_perpendicular = abs(producto_punto) < 1e-6
    
    if es_perpendicular:
        interpretacion = "El gradiente es PERPENDICULAR a la superficie de nivel"
    else:
        interpretacion = f"El gradiente NO es perfectamente perpendicular (ángulo: {angulo:.2f}°)"
    
    return {
        'producto_punto': producto_punto,
        'es_perpendicular': es_perpendicular,
        'angulo': angulo,
        'interpretacion': interpretacion
    }

def verificar_gradiente_unitario(x, y, z):
    """
    Verifica que el gradiente tenga magnitud 1 (propiedad de la función distancia).
    
    Args:
        x, y, z: Coordenadas del punto
    
    Returns:
        dict: {
            'magnitud': float,
            'es_unitario': bool,
            'interpretacion': str
        }
    """
    mag = magnitud_gradiente(x, y, z)
    es_unitario = abs(mag - 1.0) < 1e-6
    
    if mag == 0:
        interpretacion = "El gradiente es cero (en el origen - punto singular)"
    elif es_unitario:
        interpretacion = "El gradiente es UNITARIO (magnitud = 1)"
    else:
        interpretacion = f"El gradiente NO es unitario (magnitud = {mag:.6f})"
    
    return {
        'magnitud': mag,
        'es_unitario': es_unitario,
        'interpretacion': interpretacion
    }

def gradiente_proyeccion(x, y, z, dx, dy, dz):
    """
    Calcula la proyección del gradiente en una dirección dada.
    Útil para entender cómo afecta un error en cierta dirección.
    
    Args:
        x, y, z: Coordenadas del punto
        dx, dy, dz: Vector dirección (puede ser un error)
    
    Returns:
        float: Proyección del gradiente en la dirección dada
    """
    grad = gradiente(x, y, z)
    direccion = np.array([dx, dy, dz])
    norm_direccion = np.linalg.norm(direccion)
    
    if norm_direccion == 0:
        return 0.0
    
    proyeccion = np.dot(grad, direccion / norm_direccion)
    return proyeccion

def gradiente_en_direccion(x, y, z, dx, dy, dz):
    """
    Calcula la derivada direccional en la dirección dada.
    Equivalente a la proyección del gradiente en esa dirección.

    """
    return gradiente_proyeccion(x, y, z, dx, dy, dz)