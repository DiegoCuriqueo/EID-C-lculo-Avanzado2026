import numpy as np
import sympy as sp

# 1. Definición Simbólica
x_sym, y_sym, z_sym = sp.symbols('x y z')
D_sym = sp.sqrt(x_sym**2 + y_sym**2 + z_sym**2)

# Calculamos las componentes del Gradiente usando SymPy
dDx_sym = sp.diff(D_sym, x_sym)
dDy_sym = sp.diff(D_sym, y_sym)
dDz_sym = sp.diff(D_sym, z_sym)

# Convertimos el gradiente simbólico a una función numérica compacta de NumPy
num_gradiente = sp.lambdify((x_sym, y_sym, z_sym), (dDx_sym, dDy_sym, dDz_sym), 'numpy')

# 2. Funciones del Proyecto basadas en el Gradiente
def distancia(x, y, z):
    """Modelo matemático principal (Sección 5)."""
    return np.sqrt(x**2 + y**2 + z**2)

def gradiente(x, y, z):
    """ Calcula el vector gradiente de la función distancia """
    if x == 0 and y == 0 and z == 0:
        return np.array([0.0, 0.0, 0.0])
    
    # Obtenemos las componentes evaluadas numéricamente
    gx, gy, gz = num_gradiente(x, y, z)
    return np.array([float(gx), float(gy), float(gz)])

def magnitud_gradiente(x, y, z):
    """ Calcula la magnitud del gradiente """
    grad = gradiente(x, y, z)
    return np.linalg.norm(grad)

def direccion_maximo_crecimiento(x, y, z):
    """ Retorna la dirección de máximo crecimiento de la distancia. Equivalente al gradiente normalizado """
    grad = gradiente(x, y, z)
    norm = np.linalg.norm(grad)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return grad / norm