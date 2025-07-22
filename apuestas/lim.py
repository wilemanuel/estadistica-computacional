import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="🧹 Limpieza de Datos y Detección de Outliers",
    page_icon="🧹",
    layout="wide"
)

# Título principal
st.title("🧹 Limpieza de Datos y Detección de Outliers")
st.markdown("### ¡Encuentra y limpia los datos problemáticos en tu archivo CSV!")

# Sidebar con información
st.sidebar.markdown("## 📋 Guía de Limpieza")
st.sidebar.markdown("""
**¿Qué hace esta herramienta?**
- 🔍 Detecta datos faltantes
- 📊 Encuentra outliers (datos raros)
- 🧹 Sugiere cómo limpiar tus datos
- 📈 Muestra gráficos fáciles de entender
- 📋 Genera reporte completo
""")

# Función para explicar conceptos
def explicar_concepto(concepto):
    explicaciones = {
        "outliers": "🎯 **Outliers (Datos Atípicos)**: Son valores que están muy lejos del resto. Como una persona de 2.5 metros en un grupo de personas normales.",
        "datos_faltantes": "❌ **Datos Faltantes**: Son espacios vacíos en tu tabla. Como cuando no respondiste una pregunta en una encuesta.",
        "iqr": "📊 **Método IQR**: Usa cuartiles para encontrar datos raros. Si un dato está muy lejos del 'grupo principal', lo marca como extraño.",
        "zscore": "📈 **Método Z-Score**: Mide qué tan 'normal' es un dato. Si está a más de 3 desviaciones del promedio, es raro.",
        "duplicados": "👥 **Datos Duplicados**: Filas que se repiten exactamente igual. Como tener la misma persona dos veces en una lista.",
        "limpieza": "🧹 **Limpieza de Datos**: Es como ordenar tu cuarto. Quitas lo que no sirve y organizas lo que sí."
    }
    return explicaciones.get(concepto, "")

# Función para leer CSV con múltiples codificaciones
def leer_csv_con_codificacion(archivo):
    codificaciones = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for codificacion in codificaciones:
        try:
            archivo.seek(0)
            df = pd.read_csv(archivo, encoding=codificacion)
            st.success(f"✅ Archivo leído correctamente con codificación: {codificacion}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    try:
        archivo.seek(0)
        df = pd.read_csv(archivo, encoding='utf-8', errors='ignore')
        st.warning("⚠️ Se leyó el archivo ignorando algunos caracteres especiales")
        return df
    except:
        raise Exception("No se pudo leer el archivo con ninguna codificación común")

# Función para detectar outliers con IQR
def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers, limite_inferior, limite_superior

# Función para detectar outliers con Z-Score
def detectar_outliers_zscore(df, columna, umbral=3):
    z_scores = np.abs(stats.zscore(df[columna].dropna()))
    outliers = df[z_scores > umbral]
    return outliers, umbral

# Función para generar reporte de limpieza
def generar_reporte_limpieza(df):
    reporte = {}
    
    # Información general
    reporte['filas_totales'] = len(df)
    reporte['columnas_totales'] = len(df.columns)
    reporte['celdas_totales'] = len(df) * len(df.columns)
    
    # Datos faltantes
    reporte['datos_faltantes'] = df.isnull().sum().sum()
    reporte['porcentaje_faltantes'] = (reporte['datos_faltantes'] / reporte['celdas_totales']) * 100
    
    # Duplicados
    reporte['filas_duplicadas'] = df.duplicated().sum()
    reporte['porcentaje_duplicados'] = (reporte['filas_duplicadas'] / reporte['filas_totales']) * 100
    
    # Columnas con problemas
    reporte['columnas_con_faltantes'] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    
    return reporte

# Subir archivo
uploaded_file = st.file_uploader(
    "📁 Sube tu archivo CSV aquí", 
    type=['csv'],
    help="Arrastra tu archivo CSV o haz clic para seleccionarlo"
)

if uploaded_file is not None:
    try:
        # Leer el archivo
        df_original = leer_csv_con_codificacion(uploaded_file)
        df = df_original.copy()
        
        # Información básica
        st.success("✅ ¡Archivo cargado exitosamente!")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total de filas", len(df))
        with col2:
            st.metric("📊 Total de columnas", len(df.columns))
        with col3:
            st.metric("🔢 Columnas numéricas", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📝 Columnas de texto", len(df.select_dtypes(include=['object']).columns))
        
        # Generar reporte de limpieza
        reporte = generar_reporte_limpieza(df)
        
        # Mostrar vista previa
        st.subheader("👀 Vista previa de tus datos")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Tabs principales
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Reporte General", 
            "❌ Datos Faltantes", 
            "🎯 Outliers (Datos Atípicos)", 
            "🧹 Limpieza Automática",
            "📋 Reporte Final"
        ])
        
        with tab1:
            st.subheader("📊 Reporte General de Calidad")
            
            # Métricas de calidad
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "❌ Datos Faltantes", 
                    f"{reporte['datos_faltantes']:,}",
                    f"{reporte['porcentaje_faltantes']:.1f}% del total"
                )
                
                st.metric(
                    "👥 Filas Duplicadas", 
                    f"{reporte['filas_duplicadas']:,}",
                    f"{reporte['porcentaje_duplicados']:.1f}% del total"
                )
            
            with col2:
                # Gráfico de calidad general
                fig = go.Figure(data=[
                    go.Bar(
                        name='Datos Completos',
                        x=['Calidad de Datos'],
                        y=[100 - reporte['porcentaje_faltantes']],
                        marker_color='green'
                    ),
                    go.Bar(
                        name='Datos Faltantes',
                        x=['Calidad de Datos'],
                        y=[reporte['porcentaje_faltantes']],
                        marker_color='red'
                    )
                ])
                
                fig.update_layout(
                    title='Calidad General de los Datos',
                    yaxis_title='Porcentaje',
                    barmode='stack',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Calidad por columna
            st.subheader("📈 Calidad por Columna")
            
            calidad_columnas = []
            for columna in df.columns:
                faltantes = df[columna].isnull().sum()
                porcentaje_faltantes = (faltantes / len(df)) * 100
                porcentaje_completos = 100 - porcentaje_faltantes
                
                calidad_columnas.append({
                    'Columna': columna,
                    'Datos Completos': int(len(df) - faltantes),
                    'Datos Faltantes': int(faltantes),
                    '% Completos': round(porcentaje_completos, 1),
                    '% Faltantes': round(porcentaje_faltantes, 1),
                    'Tipo de Dato': str(df[columna].dtype)
                })
            
            df_calidad = pd.DataFrame(calidad_columnas)
            st.dataframe(df_calidad, use_container_width=True)
            
            # Interpretación
            st.info(explicar_concepto("datos_faltantes"))
            st.info(explicar_concepto("duplicados"))
        
        with tab2:
            st.subheader("❌ Análisis de Datos Faltantes")
            
            if reporte['datos_faltantes'] > 0:
                # Mapa de calor de datos faltantes
                st.markdown("### 🔥 Mapa de Calor de Datos Faltantes")
                st.markdown("**¿Qué muestra?** Las áreas rojas son donde faltan datos. Las azules donde están completos.")
                
                # Crear matriz de valores faltantes
                missing_matrix = df.isnull().astype(int)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(missing_matrix, cbar=True, cmap='RdYlBu_r', ax=ax)
                ax.set_title('Mapa de Datos Faltantes (Rojo = Falta, Azul = Completo)')
                st.pyplot(fig)
                
                # Estadísticas por columna
                st.markdown("### 📊 Estadísticas Detalladas")
                
                columnas_problematicas = []
                for columna, faltantes in reporte['columnas_con_faltantes'].items():
                    porcentaje = (faltantes / len(df)) * 100
                    
                    # Determinar nivel de problema
                    if porcentaje < 5:
                        nivel = "🟢 Mínimo"
                    elif porcentaje < 20:
                        nivel = "🟡 Moderado"
                    elif porcentaje < 50:
                        nivel = "🟠 Alto"
                    else:
                        nivel = "🔴 Crítico"
                    
                    columnas_problematicas.append({
                        'Columna': columna,
                        'Datos Faltantes': faltantes,
                        'Porcentaje': f"{porcentaje:.1f}%",
                        'Nivel de Problema': nivel,
                        'Recomendación': 'Eliminar columna' if porcentaje > 70 else 'Rellenar valores' if porcentaje > 30 else 'Eliminar filas'
                    })
                
                df_problematicas = pd.DataFrame(columnas_problematicas)
                st.dataframe(df_problematicas, use_container_width=True)
                
                # Recomendaciones
                st.markdown("### 💡 Recomendaciones para Datos Faltantes")
                st.markdown("""
                - **🟢 Mínimo (< 5%)**: Eliminar las filas con datos faltantes
                - **🟡 Moderado (5-20%)**: Rellenar con promedio, mediana o moda
                - **🟠 Alto (20-50%)**: Considerar técnicas avanzadas de imputación
                - **🔴 Crítico (> 50%)**: Evaluar si eliminar la columna completa
                """)
                
            else:
                st.success("🎉 ¡Excelente! No tienes datos faltantes en tu archivo.")
        
        with tab3:
            st.subheader("🎯 Detección de Outliers (Datos Atípicos)")
            
            st.info(explicar_concepto("outliers"))
            
            # Obtener columnas numéricas
            columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(columnas_numericas) > 0:
                # Selector de columna
                columna_analizar = st.selectbox(
                    "🔍 Selecciona una columna para analizar outliers:",
                    columnas_numericas
                )
                
                if columna_analizar:
                    # Selector de método
                    metodo = st.radio(
                        "📊 Selecciona el método de detección:",
                        ["IQR (Recomendado)", "Z-Score"],
                        help="IQR es más robusto, Z-Score es más sensible"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if metodo == "IQR (Recomendado)":
                            st.info(explicar_concepto("iqr"))
                            
                            # Detectar outliers con IQR
                            outliers, limite_inf, limite_sup = detectar_outliers_iqr(df, columna_analizar)
                            
                            st.metric("🎯 Outliers detectados", len(outliers))
                            st.metric("📉 Límite inferior", f"{limite_inf:.2f}")
                            st.metric("📈 Límite superior", f"{limite_sup:.2f}")
                            
                        else:
                            st.info(explicar_concepto("zscore"))
                            
                            # Detectar outliers con Z-Score
                            outliers, umbral = detectar_outliers_zscore(df, columna_analizar)
                            
                            st.metric("🎯 Outliers detectados", len(outliers))
                            st.metric("📊 Umbral Z-Score", f"{umbral}")
                    
                    with col2:
                        # Gráfico de caja
                        fig = px.box(df, y=columna_analizar, title=f'Detección de Outliers: {columna_analizar}')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar outliers detectados
                    if len(outliers) > 0:
                        st.subheader("🔍 Outliers Detectados")
                        
                        # Mostrar estadísticas de outliers
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Total de outliers", len(outliers))
                        with col2:
                            st.metric("📈 Valor máximo outlier", f"{outliers[columna_analizar].max():.2f}")
                        with col3:
                            st.metric("📉 Valor mínimo outlier", f"{outliers[columna_analizar].min():.2f}")
                        
                        # Tabla de outliers
                        st.dataframe(outliers, use_container_width=True)
                        
                        # Gráfico de dispersión
                        st.subheader("📈 Visualización de Outliers")
                        
                        # Crear gráfico de dispersión
                        fig = px.scatter(
                            df, 
                            x=df.index, 
                            y=columna_analizar, 
                            title=f'Outliers en {columna_analizar}',
                            labels={'x': 'Índice de Fila', 'y': columna_analizar}
                        )
                        
                        # Destacar outliers
                        fig.add_scatter(
                            x=outliers.index,
                            y=outliers[columna_analizar],
                            mode='markers',
                            marker=dict(color='red', size=10),
                            name='Outliers'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recomendaciones
                        st.subheader("💡 ¿Qué hacer con los outliers?")
                        porcentaje_outliers = (len(outliers) / len(df)) * 100
                        
                        if porcentaje_outliers < 1:
                            st.success(f"✅ Solo {porcentaje_outliers:.1f}% son outliers. Puedes eliminarlos de forma segura.")
                        elif porcentaje_outliers < 5:
                            st.warning(f"⚠️ {porcentaje_outliers:.1f}% son outliers. Considera si son errores o datos válidos.")
                        else:
                            st.error(f"🔴 {porcentaje_outliers:.1f}% son outliers. Revisa si hay problemas en la recolección de datos.")
                    
                    else:
                        st.success("🎉 ¡Excelente! No se detectaron outliers en esta columna.")
                
                # Análisis de todas las columnas
                st.subheader("📊 Resumen de Outliers en Todas las Columnas")
                
                resumen_outliers = []
                for columna in columnas_numericas:
                    outliers_iqr, _, _ = detectar_outliers_iqr(df, columna)
                    outliers_zscore, _ = detectar_outliers_zscore(df, columna)
                    
                    resumen_outliers.append({
                        'Columna': columna,
                        'Outliers IQR': len(outliers_iqr),
                        'Outliers Z-Score': len(outliers_zscore),
                        '% Outliers IQR': f"{(len(outliers_iqr)/len(df)*100):.1f}%",
                        '% Outliers Z-Score': f"{(len(outliers_zscore)/len(df)*100):.1f}%"
                    })
                
                df_resumen = pd.DataFrame(resumen_outliers)
                st.dataframe(df_resumen, use_container_width=True)
                
            else:
                st.warning("😔 No se encontraron columnas numéricas para analizar outliers.")
        
        with tab4:
            st.subheader("🧹 Limpieza Automática de Datos")
            
            st.info(explicar_concepto("limpieza"))
            
            # Opciones de limpieza
            st.markdown("### ⚙️ Opciones de Limpieza")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🗑️ Eliminar Datos**")
                eliminar_duplicados = st.checkbox("Eliminar filas duplicadas", value=True)
                eliminar_filas_vacias = st.checkbox("Eliminar filas completamente vacías", value=True)
                
                # Umbral para eliminar filas con muchos faltantes
                umbral_faltantes = st.slider(
                    "Eliminar filas con más de X% datos faltantes:",
                    min_value=0, max_value=100, value=50, step=5
                )
                
            with col2:
                st.markdown("**🔧 Rellenar Datos Faltantes**")
                rellenar_numericos = st.selectbox(
                    "Rellenar columnas numéricas con:",
                    ["No rellenar", "Promedio", "Mediana", "Moda", "Cero"]
                )
                
                rellenar_texto = st.selectbox(
                    "Rellenar columnas de texto con:",
                    ["No rellenar", "Moda", "Texto personalizado"]
                )
                
                if rellenar_texto == "Texto personalizado":
                    texto_personalizado = st.text_input("Texto para rellenar:", value="Sin datos")
            
            # Botón para limpiar
            if st.button("🧹 Limpiar Datos", type="primary"):
                df_limpio = df.copy()
                acciones_realizadas = []
                
                # Eliminar duplicados
                if eliminar_duplicados:
                    duplicados_antes = df_limpio.duplicated().sum()
                    df_limpio = df_limpio.drop_duplicates()
                    if duplicados_antes > 0:
                        acciones_realizadas.append(f"✅ Eliminadas {duplicados_antes} filas duplicadas")
                
                # Eliminar filas completamente vacías
                if eliminar_filas_vacias:
                    filas_vacias = df_limpio.isnull().all(axis=1).sum()
                    df_limpio = df_limpio.dropna(how='all')
                    if filas_vacias > 0:
                        acciones_realizadas.append(f"✅ Eliminadas {filas_vacias} filas completamente vacías")
                
                # Eliminar filas con muchos faltantes
                if umbral_faltantes < 100:
                    umbral_decimal = umbral_faltantes / 100
                    filas_antes = len(df_limpio)
                    df_limpio = df_limpio.dropna(thresh=len(df_limpio.columns) * (1 - umbral_decimal))
                    filas_eliminadas = filas_antes - len(df_limpio)
                    if filas_eliminadas > 0:
                        acciones_realizadas.append(f"✅ Eliminadas {filas_eliminadas} filas con más del {umbral_faltantes}% de datos faltantes")
                
                # Rellenar datos numéricos
                if rellenar_numericos != "No rellenar":
                    columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns
                    for columna in columnas_numericas:
                        faltantes_antes = df_limpio[columna].isnull().sum()
                        if faltantes_antes > 0:
                            if rellenar_numericos == "Promedio":
                                df_limpio[columna] = df_limpio[columna].fillna(df_limpio[columna].mean())
                            elif rellenar_numericos == "Mediana":
                                df_limpio[columna] = df_limpio[columna].fillna(df_limpio[columna].median())
                            elif rellenar_numericos == "Moda":
                                df_limpio[columna] = df_limpio[columna].fillna(df_limpio[columna].mode().iloc[0] if not df_limpio[columna].mode().empty else 0)
                            elif rellenar_numericos == "Cero":
                                df_limpio[columna] = df_limpio[columna].fillna(0)
                            
                            acciones_realizadas.append(f"✅ Rellenados {faltantes_antes} valores faltantes en '{columna}' con {rellenar_numericos.lower()}")
                
                # Rellenar datos de texto
                if rellenar_texto != "No rellenar":
                    columnas_texto = df_limpio.select_dtypes(include=['object']).columns
                    for columna in columnas_texto:
                        faltantes_antes = df_limpio[columna].isnull().sum()
                        if faltantes_antes > 0:
                            if rellenar_texto == "Moda":
                                df_limpio[columna] = df_limpio[columna].fillna(df_limpio[columna].mode().iloc[0] if not df_limpio[columna].mode().empty else "Sin datos")
                            elif rellenar_texto == "Texto personalizado":
                                df_limpio[columna] = df_limpio[columna].fillna(texto_personalizado)
                            
                            acciones_realizadas.append(f"✅ Rellenados {faltantes_antes} valores faltantes en '{columna}'")
                
                # Mostrar resultados
                st.subheader("📊 Resultados de la Limpieza")
                
                if acciones_realizadas:
                    for accion in acciones_realizadas:
                        st.write(accion)
                    
                    # Comparación antes/después
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Antes de la limpieza:**")
                        st.metric("Filas", len(df))
                        st.metric("Datos faltantes", df.isnull().sum().sum())
                        st.metric("Duplicados", df.duplicated().sum())
                    
                    with col2:
                        st.markdown("**✨ Después de la limpieza:**")
                        st.metric("Filas", len(df_limpio))
                        st.metric("Datos faltantes", df_limpio.isnull().sum().sum())
                        st.metric("Duplicados", df_limpio.duplicated().sum())
                    
                    # Mostrar datos limpios
                    st.subheader("📄 Datos Limpios")
                    st.dataframe(df_limpio.head(10), use_container_width=True)
                    
                    # Guardar datos limpios en session state
                    st.session_state['datos_limpios'] = df_limpio
                    
                    # Botón para descargar
                    csv_limpio = df_limpio.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Datos Limpios",
                        data=csv_limpio,
                        file_name="datos_limpios.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.info("ℹ️ No se realizaron cambios. Tus datos ya están limpios.")
        
        with tab5:
            st.subheader("📋 Reporte Final de Limpieza")
            
            # Generar reporte final
            st.markdown("### 📊 Resumen Ejecutivo")
            
            # Crear métricas de resumen
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                calidad_general = 100 - reporte['porcentaje_faltantes']
                st.metric("🏆 Calidad General", f"{calidad_general:.1f}%")
            
            with col2:
                problemas_totales = reporte['datos_faltantes'] + reporte['filas_duplicadas']
                st.metric("⚠️ Problemas Detectados", f"{problemas_totales:,}")
            
            with col3:
                columnas_problematicas = len(reporte['columnas_con_faltantes'])
                st.metric("📊 Columnas con Problemas", columnas_problematicas)
            
            with col4:
                if 'datos_limpios' in st.session_state:
                    mejora = len(df) - len(st.session_state['datos_limpios'])
                    st.metric("🧹 Filas Eliminadas", mejora)
                else:
                    st.metric("🧹 Filas Eliminadas", "No aplicado")
            
            # Reporte detallado
            st.markdown("### 📄 Reporte Detallado")
            
            reporte_detallado = f"""
            **📊 REPORTE DE CALIDAD DE DATOS**
            
            **📈 Información General:**
            - Total de filas: {reporte['filas_totales']:,}
            - Total de columnas: {reporte['columnas_totales']:,}
            - Total de celdas: {reporte['celdas_totales']:,}
            
            **❌ Problemas Detectados:**
            - Datos faltantes: {reporte['datos_faltantes']:,} ({reporte['porcentaje_faltantes']:.1f}%)
            - Filas duplicadas: {reporte['filas_duplicadas']:,} ({reporte['porcentaje_duplicados']:.1f}%)
            
            **📊 Análisis por Columna:**
            """
            
            for columna, faltantes in reporte['columnas_con_faltantes'].items():
                porcentaje = (faltantes / len(df)) * 100
                reporte_detallado += f"\n- {columna}: {faltantes} faltantes ({porcentaje:.1f}%)"
            
            if len(reporte['columnas_con_faltantes']) == 0:
                reporte_detallado += "\n- ✅ No hay columnas con datos faltantes"
            
            # Recomendaciones
            reporte_detallado += f"""
            
            **💡 Recomendaciones:**
            """
            
            if reporte['porcentaje_faltantes'] < 5:
                reporte_detallado += "\n- ✅ Calidad excelente. Mínimos problemas detectados."
            elif reporte['porcentaje_faltantes'] < 20:
                reporte_detallado += "\n- ⚠️ Calidad buena. Considerar limpiar datos faltantes."
            elif reporte['porcentaje_faltantes'] < 50:
                reporte_detallado += "\n- 🔴 Calidad regular. Limpieza necesaria antes del análisis."
            else:
                reporte_detallado += "\n- 💀 Calidad crítica. Revisar proceso de recolección de datos."
            
            if reporte['filas_duplicadas'] > 0:
                reporte_detallado += f"\n- 🗑️ Eliminar {reporte['filas_duplicadas']} filas duplicadas"
            
            if len(columnas_numericas) > 0:
                reporte_detallado += "\n- 🎯 Revisar outliers en columnas numéricas"
            
            st.markdown(reporte_detallado)
            
            # Gráfico de resumen
            st.markdown("### 📊 Visualización del Reporte")
            
            # Crear gráfico de pastel para mostrar distribución de problemas
            fig = go.Figure(data=[go.Pie(
                labels=['Datos Completos', 'Datos Faltantes', 'Duplicados'],
                values=[
                    reporte['celdas_totales'] - reporte['datos_faltantes'] - reporte['filas_duplicadas'],
                    reporte['datos_faltantes'],
                    reporte['filas_duplicadas']
                ],
                hole=0.3,
                marker_colors=['green', 'red', 'orange']
            )])
            
            fig.update_layout(
                title="Distribución de Calidad de Datos",
                annotations=[dict(text='Calidad<br>General', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Descargar reporte
            st.download_button(
                label="📥 Descargar Reporte Completo",
                data=reporte_detallado,
                file_name="reporte_limpieza_datos.txt",
                mime="text/plain"
            )
            
            # Conclusiones finales
            st.markdown("### 🎯 Conclusiones y Próximos Pasos")
            
            conclusiones = []
            
            if reporte['porcentaje_faltantes'] < 5 and reporte['filas_duplicadas'] == 0:
                conclusiones.append("🎉 **¡Excelente!** Tus datos están en muy buena condición.")
            elif reporte['porcentaje_faltantes'] < 20:
                conclusiones.append("✅ **Buena calidad** general, solo necesitas limpieza menor.")
            else:
                conclusiones.append("⚠️ **Necesitas limpieza** antes de hacer análisis importantes.")
            
            if len(columnas_numericas) > 0:
                conclusiones.append("📊 **Revisa los outliers** en la pestaña correspondiente.")
            
            if reporte['filas_duplicadas'] > 0:
                conclusiones.append("🗑️ **Elimina los duplicados** para evitar sesgos en tu análisis.")
            
            conclusiones.append("🧹 **Usa la limpieza automática** para mejorar la calidad de tus datos.")
            
            for conclusion in conclusiones:
                st.markdown(conclusion)
            
            # Recordatorio importante
            st.warning("⚠️ **Recordatorio importante**: Siempre revisa manualmente los cambios propuestos antes de aplicarlos a tus datos originales.")
    
    except Exception as e:
        st.error(f"😞 Hubo un error al procesar tu archivo: {str(e)}")
        st.markdown("**Posibles soluciones:**")
        st.markdown("- **Problema de codificación**: Guarda tu archivo CSV como UTF-8")
        st.markdown("- **Archivo dañado**: Verifica que tu archivo CSV no esté corrupto")
        st.markdown("- **Formato incorrecto**: Asegúrate de que sea un archivo CSV válido")
        st.markdown("- **Caracteres especiales**: Evita caracteres especiales en nombres de columnas")
        
        # Información técnica
        st.markdown("### 🔧 Información técnica:")
        st.code(f"Error específico: {str(e)}", language='text')

else:
    # Página de inicio cuando no hay archivo
    st.info("👆 Sube un archivo CSV para comenzar el análisis de limpieza")
    
    st.markdown("### 🤔 ¿Qué son los datos sucios?")
    st.markdown("""
    Los datos "sucios" son datos que tienen problemas como:
    
    - **❌ Datos faltantes**: Celdas vacías o con valores nulos
    - **👥 Duplicados**: Filas que se repiten exactamente
    - **🎯 Outliers**: Valores muy diferentes al resto (datos atípicos)
    - **📝 Inconsistencias**: Diferentes formatos para el mismo tipo de dato
    - **🔤 Errores de tipeo**: Nombres mal escritos o caracteres raros
    """)
    
    st.markdown("### 🧹 ¿Por qué limpiar los datos?")
    st.markdown("""
    - **📊 Análisis más precisos**: Resultados más confiables
    - **🎯 Mejores decisiones**: Conclusiones basadas en datos correctos
    - **⚡ Procesamiento más rápido**: Menos datos problemáticos
    - **🔍 Patrones más claros**: Tendencias más fáciles de identificar
    """)
    
    st.markdown("### 📋 Ejemplo de datos que necesitan limpieza:")
    
    ejemplo_sucio = """Nombre,Edad,Salario,Ciudad
Juan,25,45000,Madrid
María,,55000,Barcelona
Pedro,35,65000,Madrid
Ana,28,50000,
Juan,25,45000,Madrid
Carlos,200,70000,Valencia"""
    
    st.code(ejemplo_sucio, language='csv')
    
    st.markdown("**🔍 Problemas en este ejemplo:**")
    st.markdown("- Edad faltante para María")
    st.markdown("- Ciudad faltante para Ana")
    st.markdown("- Juan está duplicado")
    st.markdown("- Carlos tiene 200 años (posible outlier)")
    
    st.markdown("### 🎯 Lo que esta herramienta hace por ti:")
    st.markdown("""
    - **🔍 Detecta automáticamente** todos los problemas
    - **📊 Muestra gráficos** fáciles de entender
    - **🧹 Limpia automáticamente** con un clic
    - **📋 Genera reportes** detallados
    - **💡 Da recomendaciones** específicas
    """)

# Footer
st.markdown("---")
st.markdown("🎯 **Recuerda**: Datos limpios = Análisis confiables = Mejores decisiones")