"""
Módulo para visualizaciones 3D con Matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from src.utils.constants import COLOR_RUTA_REAL, COLOR_RUTA_MEDIDA, COLOR_BASE

def plot_trayectoria_3d(df_resultados, mostrar_errores=True, guardar=True):
    """
    Visualiza la trayectoria en 3D con la posición real y medida.
    
    Args:
        df_resultados: DataFrame con resultados
        mostrar_errores: Si muestra líneas de error
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Posiciones reales
    ax.scatter(
        df_resultados['x'], df_resultados['y'], df_resultados['z'],
        color='green', s=30, label='Posición Real', alpha=0.7
    )
    ax.plot(
        df_resultados['x'], df_resultados['y'], df_resultados['z'],
        color='green', linewidth=2, alpha=0.5
    )
    
    # Posiciones medidas (con error)
    if 'dx' in df_resultados.columns and 'dy' in df_resultados.columns:
        ax.scatter(
            df_resultados['x'] + df_resultados['dx'],
            df_resultados['y'] + df_resultados['dy'],
            df_resultados['z'] + df_resultados['dz'],
            color='red', s=30, label='Posición Medida', alpha=0.7
        )
        ax.plot(
            df_resultados['x'] + df_resultados['dx'],
            df_resultados['y'] + df_resultados['dy'],
            df_resultados['z'] + df_resultados['dz'],
            color='red', linewidth=2, alpha=0.5, linestyle='--'
        )
    
    # Mostrar errores como líneas
    if mostrar_errores and 'dx' in df_resultados.columns:
        for _, row in df_resultados.iterrows():
            ax.plot(
                [row['x'], row['x'] + row['dx']],
                [row['y'], row['y'] + row['dy']],
                [row['z'], row['z'] + row['dz']],
                color='orange', linewidth=1, alpha=0.3
            )
    
    # Estación base (origen)
    ax.scatter([0], [0], [0], color='red', s=200, marker='*', label='Estación Base')
    
    # Configuración
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trayectoria del Dron en 3D')
    ax.legend()
    
    # Igualar escalas
    max_range = max([
        df_resultados['x'].max() - df_resultados['x'].min(),
        df_resultados['y'].max() - df_resultados['y'].min(),
        df_resultados['z'].max() - df_resultados['z'].min()
    ]) / 2
    
    if max_range > 0:
        mid_x = (df_resultados['x'].max() + df_resultados['x'].min()) / 2
        mid_y = (df_resultados['y'].max() + df_resultados['y'].min()) / 2
        mid_z = (df_resultados['z'].max() + df_resultados['z'].min()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/trayectoria_3d.png', dpi=300, bbox_inches='tight')
    
    return fig

def plot_superficies_nivel(x, y, z, D_val, guardar=True):
    """
    Visualiza superficies de nivel (esferas) y gradientes.
    
    Args:
        x, y, z: Coordenadas del punto
        D_val: Valor de la distancia (radio de la esfera)
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear esfera de nivel
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    X = D_val * np.outer(np.cos(u), np.sin(v))
    Y = D_val * np.outer(np.sin(u), np.sin(v))
    Z = D_val * np.outer(np.ones_like(u), np.cos(v))
    
    # Graficar esfera
    ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', label='Superficie de nivel')
    
    # Punto del dron
    ax.scatter([x], [y], [z], color='red', s=100, label='Dron')
    
    # Gradiente (flecha radial)
    grad = np.array([x, y, z]) / D_val if D_val > 0 else np.array([0, 0, 0])
    ax.quiver(x, y, z, grad[0], grad[1], grad[2], 
              length=0.5*D_val if D_val > 0 else 1, 
              color='blue', label='Gradiente')
    
    # Origen
    ax.scatter([0], [0], [0], color='black', s=150, marker='*', label='Estación Base')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Superficie de nivel D = {D_val:.2f} m y Gradiente')
    ax.legend()
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/superficie_nivel.png', dpi=300, bbox_inches='tight')
    
    return fig

def plot_gradientes(df_resultados, guardar=True):
    """
    Visualiza los gradientes en toda la trayectoria.
    
    Args:
        df_resultados: DataFrame con resultados (debe tener 'grad_x', 'grad_y', 'grad_z')
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Verificar si tiene gradientes calculados
    if 'grad_x' in df_resultados.columns:
        # Trayectoria real
        ax.plot(df_resultados['x'], df_resultados['y'], df_resultados['z'],
                color='green', linewidth=2, alpha=0.5, label='Trayectoria')
        
        # Gradientes como flechas
        for _, row in df_resultados.iterrows():
            ax.quiver(
                row['x'], row['y'], row['z'],
                row['grad_x'], row['grad_y'], row['grad_z'],
                length=0.3,
                color='blue',
                alpha=0.6
            )
    
    # Estación base
    ax.scatter([0], [0], [0], color='red', s=200, marker='*', label='Estación Base')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Gradientes a lo largo de la trayectoria')
    ax.legend()
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/gradientes_3d.png', dpi=300, bbox_inches='tight')
    
    return fig

def plot_diferencial_vs_exacto_3d(df_resultados, guardar=True):
    """
    Visualiza en 3D la comparación entre error exacto y estimado.
    
    Args:
        df_resultados: DataFrame con resultados
        guardar: Si guarda la figura en archivo
    
    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    if guardar:
        os.makedirs('data/resultados', exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Barras de error en 3D
    x_pos = df_resultados.index
    y_pos = df_resultados['distancia_real']
    
    ax.bar3d(
        x_pos, y_pos, 0,
        dx=0.5, dy=0.5, dz=df_resultados['error_exacto'],
        color='blue', alpha=0.5, label='Error Exacto'
    )
    
    ax.bar3d(
        x_pos + 0.5, y_pos, 0,
        dx=0.5, dy=0.5, dz=df_resultados['error_estimado'],
        color='red', alpha=0.5, label='Error Estimado'
    )
    
    ax.set_xlabel('Punto')
    ax.set_ylabel('Distancia (m)')
    ax.set_zlabel('Error (m)')
    ax.set_title('Comparación 3D: Error Exacto vs Estimado')
    ax.legend()
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig('data/resultados/comparacion_3d.png', dpi=300, bbox_inches='tight')
    
    return fig