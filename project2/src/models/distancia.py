import numpy as np
import sympy as sp

# 1. Definición Simbólica
x_sym, y_sym, z_sym = sp.symbols('x y z')
D_sym = sp.sqrt(x_sym**2 + y_sym**2 + z_sym**2)

# Calculamos las derivadas parciales solicitadas (Sección 6.3)
dDx_sym = sp.diff(D_sym, x_sym)
dDy_sym = sp.diff(D_sym, y_sym)
dDz_sym = sp.diff(D_sym, z_sym)

# Convertimos a funciones numéricas de NumPy usando lambdify para la simulación
num_derivadas = sp.lambdify((x_sym, y_sym, z_sym), (dDx_sym, dDy_sym, dDz_sym), 'numpy')

# 2. Funciones del Proyecto
def distancia(x, y, z):
    """ Modelo matemático """
    return np.sqrt(x**2 + y**2 + z**2)

def distancia_entre_puntos(x1, y1, z1, x2, y2, z2):
    """Distancia entre dos puntos (Útil para extensiones o variantes)."""
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def derivadas_parciales(x, y, z):
    """ Calcula las tres derivadas parciales de forma normal usando SymPy """
    if x == 0 and y == 0 and z == 0:
        return 0.0, 0.0, 0.0
    
    dDx, dDy, dDz = num_derivadas(x, y, z)
    return float(dDx), float(dDy), float(dDz)

def diferencial_total(x, y, z, dx, dy, dz):
    """ Calcula el Diferencial Total """
    dDx, dDy, dDz = derivadas_parciales(x, y, z)
    return dDx * dx + dDy * dy + dDz * dz

def error_exacto(x, y, z, dx, dy, dz):
    """ Diferencia real entre la posición medida y la real """
    return distancia(x + dx, y + dy, z + dz) - distancia(x, y, z)

def comparacion_errores(x, y, z, dx, dy, dz):
    """ Compara el error exacto vs la estimación lineal """
    exacto = error_exacto(x, y, z, dx, dy, dz)
    estimado = diferencial_total(x, y, z, dx, dy, dz)
    return {
        'error_exacto': exacto,
        'error_estimado': estimado,
        'diferencia_de_error': exacto - estimado,
        'error_relativo_pct': ((exacto - estimado)/exacto*100) if exacto != 0 else 0
    }

def analisis_sensibilidad_punto(x, y, z, dx, dy, dz):
    """ Analiza el impacto relativo de cada coordenada en el error """
    dDx, dDy, dDz = derivadas_parciales(x, y, z)
    error_estimado = diferencial_total(x, y, z, dx, dy, dz)
    
    contrib_x = dDx * dx
    contrib_y = dDy * dy
    contrib_z = dDz * dz
    
    if error_estimado == 0:
        return {'contrib_x': contrib_x, 'contrib_y': contrib_y, 'contrib_z': contrib_z, 
                'contrib_x_pct': 0, 'contrib_y_pct': 0, 'contrib_z_pct': 0}
    
    return {
        'contrib_x': contrib_x,
        'contrib_y': contrib_y,
        'contrib_z': contrib_z,
        'contrib_x_pct': abs(contrib_x / error_estimado * 100),
        'contrib_y_pct': abs(contrib_y / error_estimado * 100),
        'contrib_z_pct': abs(contrib_z / error_estimado * 100)
    }