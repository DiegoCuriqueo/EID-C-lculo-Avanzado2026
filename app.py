"""
Aplicacion Streamlit para analisis de propagacion de errores en drones.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Configurar rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==================== IMPORTACIONES ====================
from src.data_processing.loader import cargar_csv, validar_coordenadas
from src.data_processing.converter import convertir_coordenadas_relativas
from src.data_processing.errors import (
    generar_errores,
    generar_errores_gps_real,
    generar_errores_con_deriva
)

from src.models.distancia import distancia, derivadas_parciales

from src.analysis.error_propagation import (
    analizar_propagacion_errores,
    calcular_estadisticas_errores,
    calcular_errores_por_punto
)
from src.analysis.sensitivity import (
    analisis_sensibilidad,
    analisis_por_distancia
)
from src.analysis.comparison import (
    comparar_errores,
    metricas_precision,
    tabla_comparativa,
    generar_resumen_comparativo
)

from src.visualization.map_plotter import crear_mapa_interactivo
from src.visualization.error_plots import graficar_todas_visualizaciones
from src.visualization._3d_plotter import plot_trayectoria_3d

from src.utils.constants import (
    TEMUCO_LAT,
    TEMUCO_LON,
    STREAMLIT_TITLE,
    ERROR_GPS_ESTANDAR_STD_XY,
    ERROR_GPS_ESTANDAR_STD_Z,
    ERROR_GPS_PRECISO_STD_XY,
    ERROR_GPS_PRECISO_STD_Z,
    ERROR_GPS_IMPRECISO_STD_XY,
    ERROR_GPS_IMPRECISO_STD_Z
)
from src.utils.helpers import generar_csv_ejemplo

# ==================== CONFIGURACION DE PAGINA ====================
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== TITULO ====================
st.title(STREAMLIT_TITLE)
st.markdown("---")

# ==================== INICIALIZAR ESTADO DE SESION ====================
if 'analisis_realizado' not in st.session_state:
    st.session_state.analisis_realizado = False
if 'resultados_analysis' not in st.session_state:
    st.session_state.resultados_analysis = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Configuracion")
    
    # ===== Carga de archivo =====
    st.subheader("Datos de entrada")
    
    opcion_datos = st.radio(
        "Selecciona fuente de datos:",
        ["Cargar archivo CSV", "Usar datos de ejemplo"]
    )
    
    uploaded_file = None
    
    if opcion_datos == "Cargar archivo CSV":
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV con coordenadas",
            type=['csv'],
            help="El archivo debe tener columnas: lat, lon, altitud (opcional), tiempo (opcional)"
        )
        
        if uploaded_file is not None:
            st.session_state.df_original = pd.read_csv(uploaded_file)
            st.success(f"Datos cargados: {len(st.session_state.df_original)} puntos")
    else:
        # Generar datos de ejemplo
        if st.button("Generar datos de ejemplo"):
            with st.spinner("Generando datos de ejemplo..."):
                ruta = generar_csv_ejemplo()
                st.session_state.df_original = cargar_csv(ruta)
                st.success(f"Datos de ejemplo generados: {len(st.session_state.df_original)} puntos")
    
    st.markdown("---")
    
    # ===== Configuracion de errores =====
    st.subheader("Errores de medicion")
    
    tipo_error = st.selectbox(
        "Tipo de error GPS:",
        ["Estandar (σ=0.5m)", "Preciso (σ=0.1m)", "Impreciso (σ=2m)", "Con deriva temporal", "Personalizado"]
    )
    
    if tipo_error == "Estandar (σ=0.5m)":
        std_xy = ERROR_GPS_ESTANDAR_STD_XY
        std_z = ERROR_GPS_ESTANDAR_STD_Z
        tipo_error_key = 'estandar'
    elif tipo_error == "Preciso (σ=0.1m)":
        std_xy = ERROR_GPS_PRECISO_STD_XY
        std_z = ERROR_GPS_PRECISO_STD_Z
        tipo_error_key = 'preciso'
    elif tipo_error == "Impreciso (σ=2m)":
        std_xy = ERROR_GPS_IMPRECISO_STD_XY
        std_z = ERROR_GPS_IMPRECISO_STD_Z
        tipo_error_key = 'impreciso'
    elif tipo_error == "Con deriva temporal":
        col1, col2 = st.columns(2)
        with col1:
            drift_xy = st.slider("Deriva XY (m/paso)", 0.0, 0.1, 0.01, 0.001)
        with col2:
            drift_z = st.slider("Deriva Z (m/paso)", 0.0, 0.05, 0.005, 0.001)
        tipo_error_key = 'deriva'
    else:  # Personalizado
        col1, col2, col3 = st.columns(3)
        with col1:
            std_xy = st.slider("σ para x,y (m)", 0.1, 5.0, 0.5, 0.1)
        with col2:
            std_z = st.slider("σ para z (m)", 0.05, 2.0, 0.2, 0.05)
        with col3:
            correlacion = st.slider("Correlacion", 0.0, 1.0, 0.0, 0.1)
        tipo_error_key = 'personalizado'
    
    st.markdown("---")
    
    # ===== Boton de analisis =====
    analizar_btn = st.button("ANALIZAR", use_container_width=True, type="primary")

# ==================== FUNCION DE ANALISIS ====================
def realizar_analisis(df, std_xy, std_z, tipo_error_key, **kwargs):
    """
    Ejecuta todo el analisis con los datos proporcionados.
    """
    with st.spinner("Procesando datos..."):
        # Convertir coordenadas
        df_convertido, base_lat, base_lon = convertir_coordenadas_relativas(df)
        
        # Generar errores
        n = len(df)
        if tipo_error_key == 'estandar':
            dx, dy, dz = generar_errores_gps_real(n, 'estandar')
        elif tipo_error_key == 'preciso':
            dx, dy, dz = generar_errores_gps_real(n, 'preciso')
        elif tipo_error_key == 'impreciso':
            dx, dy, dz = generar_errores_gps_real(n, 'impreciso')
        elif tipo_error_key == 'deriva':
            dx, dy, dz = generar_errores_con_deriva(
                n, 
                drift_xy=kwargs.get('drift_xy', 0.01),
                drift_z=kwargs.get('drift_z', 0.005)
            )
        else:  # personalizado
            dx, dy, dz = generar_errores(
                n, 
                std_xy=std_xy, 
                std_z=std_z,
                correlacion=kwargs.get('correlacion', 0)
            )
        
        # Analizar propagacion
        resultados = analizar_propagacion_errores(df_convertido, dx, dy, dz)
        
        # Estadisticas
        stats = calcular_estadisticas_errores(resultados)
        
        # Errores por punto
        resultados_punto = calcular_errores_por_punto(resultados)
        
        # Sensibilidad
        sensibilidad, resultados_sens = analisis_sensibilidad(resultados)
        
        # Analisis por distancia
        precision_rangos = analisis_por_distancia(resultados)
        
        # Metricas de precision
        metricas = metricas_precision(resultados)
        
        # Resumen comparativo
        resumen = generar_resumen_comparativo(resultados)
        
        return {
            'resultados': resultados,
            'resultados_punto': resultados_punto,
            'stats': stats,
            'sensibilidad': sensibilidad,
            'precision_rangos': precision_rangos,
            'metricas': metricas,
            'resumen': resumen,
            'base_lat': base_lat,
            'base_lon': base_lon,
            'df_convertido': df_convertido,
            'df_original': df
        }

# ==================== LOGICA PRINCIPAL ====================

# Verificar si se presiono el boton de analisis
if analizar_btn:
    if st.session_state.df_original is not None:
        # Ejecutar analisis
        kwargs = {}
        if tipo_error_key == 'deriva':
            kwargs['drift_xy'] = drift_xy
            kwargs['drift_z'] = drift_z
        elif tipo_error_key == 'personalizado':
            kwargs['correlacion'] = correlacion
        
        st.session_state.resultados_analysis = realizar_analisis(
            st.session_state.df_original, 
            std_xy, 
            std_z, 
            tipo_error_key,
            **kwargs
        )
        st.session_state.analisis_realizado = True
        st.rerun()
    else:
        st.warning("Primero debes cargar un archivo CSV o generar datos de ejemplo")

# Mostrar resultados si ya se realizo el analisis
if st.session_state.analisis_realizado and st.session_state.resultados_analysis is not None:
    
    resultados_analysis = st.session_state.resultados_analysis
    
    resultados = resultados_analysis['resultados']
    resultados_punto = resultados_analysis['resultados_punto']
    stats = resultados_analysis['stats']
    sensibilidad = resultados_analysis['sensibilidad']
    precision_rangos = resultados_analysis['precision_rangos']
    metricas = resultados_analysis['metricas']
    resumen = resultados_analysis['resumen']
    base_lat = resultados_analysis['base_lat']
    base_lon = resultados_analysis['base_lon']
    df_convertido = resultados_analysis['df_convertido']
    df_original = resultados_analysis['df_original']
    
    # ===== MOSTRAR RESULTADOS =====
    
    # Tabs para organizar
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resumen",
        "Graficos",
        "Mapa",
        "Analisis detallado",
        "Sensibilidad",
        "Datos"
    ])
    
    # ===== TAB 1: RESUMEN =====
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Error exacto medio",
            f"{stats['Error exacto medio'].iloc[0]:.4f} m"
        )
    
    with col2:
        st.metric(
            "Error estimado medio",
            f"{stats['Error estimado medio'].iloc[0]:.4f} m"
        )
    
    with col3:
        st.metric(
            "Error de aprox. medio",
            f"{stats['Error de aproximación medio'].iloc[0]:.4f} m"  # ✅ Corregido
        )
    
    with col4:
        correlacion_val = stats['Correlación exacto-estimado'].iloc[0]  # ✅ Corregido
        st.metric(
            "Correlacion",
            f"{correlacion_val:.4f}",
            delta=None
        )
        
        st.markdown("---")
        
        # Resumen ejecutivo
        st.subheader("Resumen ejecutivo")
        st.info(resumen['conclusion'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Correlacion",
                f"{resumen['correlacion']:.4f}",
                resumen['correlacion_texto']
            )
        with col2:
            st.metric(
                "Error medio",
                f"{resumen['error_medio']:.4f} m",
                resumen['precision_texto']
            )
        with col3:
            st.metric(
                "Error maximo",
                f"{resumen['error_max']:.4f} m"
            )
        
        st.markdown("---")
        
        # Metricas de precision
        st.subheader("Metricas de precision")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metricas['MAE']:.4f} m")
        with col2:
            st.metric("RMSE", f"{metricas['RMSE']:.4f} m")
        with col3:
            st.metric("R2", f"{metricas['R2']:.4f}")
    
    # ===== TAB 2: GRAFICOS =====
    with tab2:
        st.subheader("Visualizacion de errores")
        
        # Grafico 1: Error exacto vs estimado
        fig1 = px.scatter(
            resultados,
            x='error_exacto',
            y='error_estimado',
            title='Error Exacto vs Error Estimado',
            labels={'error_exacto': 'Error Exacto (m)', 'error_estimado': 'Error Estimado (m)'},
            color_discrete_sequence=['blue']
        )
        
        # Agregar linea de perfecta correlacion
        min_val = min(resultados['error_exacto'].min(), resultados['error_estimado'].min())
        max_val = max(resultados['error_exacto'].max(), resultados['error_estimado'].max())
        fig1.add_scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfecta correlacion',
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Grafico 2: Contribuciones en el tiempo
        fig2 = px.line(
            resultados,
            x=resultados.index,
            y=['contrib_x', 'contrib_y', 'contrib_z'],
            title='Contribucion de cada coordenada al error',
            labels={'value': 'Contribucion (m)', 'index': 'Punto en trayectoria', 'variable': 'Coordenada'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Grafico 3: Distribucion del error
        fig3 = px.histogram(
            resultados,
            x='error_aproximacion',
            title='Distribucion del error de aproximacion',
            labels={'error_aproximacion': 'Error de aproximacion (m)', 'count': 'Frecuencia'},
            nbins=20,
            color_discrete_sequence=['green']
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Error cero")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Grafico 4: Precision vs Distancia
        fig4 = px.scatter(
            resultados,
            x='distancia_real',
            y='error_aproximacion',
            title='Precision de la aproximacion vs Distancia',
            labels={'distancia_real': 'Distancia a la base (m)', 'error_aproximacion': 'Error de aproximacion (m)'},
            color='distancia_real',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # Grafico 5: Evolucion temporal
        if 'tiempo' in df_original.columns:
            fig5 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig5.add_trace(
                go.Scatter(x=df_original['tiempo'], y=resultados['distancia_real'],
                          name='Distancia Real', line=dict(color='blue')),
                secondary_y=False
            )
            fig5.add_trace(
                go.Scatter(x=df_original['tiempo'], y=resultados['error_exacto'],
                          name='Error Exacto', line=dict(color='red', dash='dash')),
                secondary_y=True
            )
            fig5.add_trace(
                go.Scatter(x=df_original['tiempo'], y=resultados['error_estimado'],
                          name='Error Estimado', line=dict(color='orange', dash='dot')),
                secondary_y=True
            )
            
            fig5.update_layout(title='Evolucion temporal de distancia y errores')
            fig5.update_xaxes(title_text='Tiempo (s)')
            fig5.update_yaxes(title_text='Distancia (m)', secondary_y=False)
            fig5.update_yaxes(title_text='Error (m)', secondary_y=True)
            
            st.plotly_chart(fig5, use_container_width=True)
    
    # ===== TAB 3: MAPA =====
    with tab3:
        st.subheader("Ruta del dron")
        
        # Crear mapa
        mapa = crear_mapa_interactivo(
            resultados,
            base_lat,
            base_lon
        )
        
        # Mostrar mapa
        from streamlit_folium import st_folium
        st_folium(mapa, width=900, height=600)
        
        st.caption("Verde: Ruta real | Rojo: Ruta con errores | Estacion base")
    
    # ===== TAB 4: ANALISIS DETALLADO =====
    with tab4:
        st.subheader("Analisis detallado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Punto con mayor error
            max_error = resultados.loc[resultados['error_exacto'].abs().idxmax()]
            st.info("Punto con mayor error")
            st.write(f"Distancia: {max_error['distancia_real']:.2f} m")
            st.write(f"Error exacto: {max_error['error_exacto']:.4f} m")
            st.write(f"Error estimado: {max_error['error_estimado']:.4f} m")
            st.write(f"Diferencia: {max_error['error_aproximacion']:.4f} m")
        
        with col2:
            # Punto con mejor aproximacion
            min_error = resultados.loc[resultados['error_aproximacion'].abs().idxmin()]
            st.success("Punto con mejor aproximacion")
            st.write(f"Distancia: {min_error['distancia_real']:.2f} m")
            st.write(f"Error exacto: {min_error['error_exacto']:.4f} m")
            st.write(f"Error estimado: {min_error['error_estimado']:.4f} m")
            st.write(f"Diferencia: {min_error['error_aproximacion']:.4f} m")
        
        st.markdown("---")
        
        # Interpretacion
        st.subheader("Interpretacion de resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ventajas de la aproximacion diferencial:**")
            st.markdown("""
            - Rapida de calcular (sin raices cuadradas)
            - Ideal para sistemas en tiempo real
            - Util para estimar errores sin recalcular
            - El gradiente indica la direccion de maximo error
            """)
        
        with col2:
            st.markdown("**Limitaciones:**")
            st.markdown("""
            - Solo precisa para errores pequeños
            - Fallan cerca del origen (no diferenciable)
            - No captura efectos de segundo orden
            - Depende de la geometria de la trayectoria
            """)
        
        st.markdown("---")
        
        # Condiciones de precision
        st.subheader("Condiciones de precision")
        st.markdown(f"""
        **La aproximacion es mas precisa cuando:**
        - Los errores son pequeños (< 1m)
        - El dron esta lejos del origen (> 100m)
        - Las coordenadas no son cercanas a cero
        - La trayectoria es suave (sin cambios bruscos)
        
        **En este analisis:**
        - Error medio: **{resumen['error_medio']:.4f} m**
        - Correlacion: **{resumen['correlacion']:.4f}**
        - Precision: **{resumen['precision_texto']}**
        """)
    
    # ===== TAB 5: SENSIBILIDAD =====
    with tab5:
        col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Sensibilidad en x",
            f"{sensibilidad['Contribución Promedio (%)'].iloc[0]:.1f}%"  # ✅ Corregido
        )
    with col2:
        st.metric(
            "Sensibilidad en y",
            f"{sensibilidad['Contribución Promedio (%)'].iloc[1]:.1f}%"  # ✅ Corregido
        )
    with col3:
        st.metric(
            "Sensibilidad en z",
            f"{sensibilidad['Contribución Promedio (%)'].iloc[2]:.1f}%"  # ✅ Corregido
        )
        
        st.markdown("---")
        
        # Tabla completa de sensibilidad
        st.subheader("Detalle de sensibilidad")
        st.dataframe(sensibilidad, use_container_width=True)
        
        st.markdown("---")
        
        # Precision por rango de distancia
        st.subheader("Precision por rango de distancia")
        st.dataframe(precision_rangos, use_container_width=True)
    
    # ===== TAB 6: DATOS =====
    with tab6:
        st.subheader("Datos completos del analisis")
        
        # Selector de columnas
        columnas_disponibles = resultados.columns.tolist()
        columnas_por_defecto = ['punto', 'distancia_real', 'error_exacto', 'error_estimado', 'error_aproximacion']
        
        columnas_mostrar = st.multiselect(
            "Selecciona columnas para visualizar",
            columnas_disponibles,
            default=[col for col in columnas_por_defecto if col in columnas_disponibles]
        )
        
        if columnas_mostrar:
            st.dataframe(resultados[columnas_mostrar], use_container_width=True)
        
        # Boton de descarga
        st.markdown("---")
        st.subheader("Descargar datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_resultados = resultados.to_csv(index=False)
            st.download_button(
                label="Descargar resultados CSV",
                data=csv_resultados,
                file_name="analisis_errores_dron.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_resumen = pd.DataFrame([resumen]).to_csv(index=False)
            st.download_button(
                label="Descargar resumen CSV",
                data=csv_resumen,
                file_name="resumen_analisis_dron.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== ESTADO INICIAL ====================
elif st.session_state.df_original is not None and not analizar_btn:
    st.info("Configura los parametros en la barra lateral y presiona 'ANALIZAR'")

else:
    # ==================== PAGINA DE INICIO ====================
    st.markdown("""
    ## Bienvenido al analizador de errores para drones
    
    ### Instrucciones:
    1. Carga un archivo CSV con coordenadas en la barra lateral
    2. Configura los errores de medicion GPS
    3. Presiona ANALIZAR para ver los resultados
    
    ### Formato del archivo CSV:
    El archivo debe tener las siguientes columnas:
    - `lat`: Latitud (grados decimales)
    - `lon`: Longitud (grados decimales)
    - `altitud`: Altura (metros) - opcional
    - `tiempo`: Tiempo (segundos) - opcional
    """)
   