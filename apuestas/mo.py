import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Storytelling: Análisis MANOVA de Servicios Básicos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Storytelling con MANOVA: Análisis de Gastos en Servicios Básicos")
st.markdown("---")

# Función para generar datos sintéticos (como ejemplo)
@st.cache_data
def generar_datos_ejemplo():
    """Genera datos sintéticos como ejemplo"""
    np.random.seed(42)
    n_hogares = 300
    
    # Definir grupos (estratos socioeconómicos)
    grupos = np.random.choice(['Bajo', 'Medio', 'Alto'], n_hogares, p=[0.4, 0.4, 0.2])
    
    # Generar gastos base según grupo
    gastos_base = {
        'Bajo': {'agua': 50, 'luz': 80, 'gas': 40, 'internet': 30, 'telefono': 25},
        'Medio': {'agua': 80, 'luz': 150, 'gas': 70, 'internet': 60, 'telefono': 45},
        'Alto': {'agua': 120, 'luz': 250, 'gas': 100, 'internet': 100, 'telefono': 70}
    }
    
    # Crear DataFrame
    data = []
    for i, grupo in enumerate(grupos):
        base = gastos_base[grupo]
        # Agregar variabilidad realista
        agua = np.random.normal(base['agua'], base['agua'] * 0.3)
        luz = np.random.normal(base['luz'], base['luz'] * 0.4)
        gas = np.random.normal(base['gas'], base['gas'] * 0.35)
        internet = np.random.normal(base['internet'], base['internet'] * 0.2)
        telefono = np.random.normal(base['telefono'], base['telefono'] * 0.25)
        
        # Asegurar valores positivos
        agua = max(20, agua)
        luz = max(30, luz)
        gas = max(15, gas)
        internet = max(10, internet)
        telefono = max(10, telefono)
        
        data.append({
            'hogar_id': i+1,
            'grupo': grupo,
            'agua': round(agua, 2),
            'luz': round(luz, 2),
            'gas': round(gas, 2),
            'internet': round(internet, 2),
            'telefono': round(telefono, 2),
            'total': round(agua + luz + gas + internet + telefono, 2)
        })
    
    return pd.DataFrame(data)

# Función para validar datos
def validar_datos(df):
    """Valida que los datos sean apropiados para MANOVA"""
    errores = []
    
    if df is None or df.empty:
        errores.append("El archivo está vacío")
        return errores
    
    # Verificar que tenga al menos 3 filas
    if len(df) < 10:
        errores.append("Se necesitan al menos 10 observaciones para un análisis confiable")
    
    # Verificar columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) < 2:
        errores.append("Se necesitan al menos 2 variables numéricas para MANOVA")
    
    return errores

# Función para limpiar nombres de columnas
def limpiar_nombres_columnas(df):
    """Limpia los nombres de columnas para evitar problemas con MANOVA"""
    df_limpio = df.copy()
    
    # Diccionario para mapear nombres originales a nombres limpios
    mapeo_nombres = {}
    
    for col in df_limpio.columns:
        # Limpiar el nombre: solo letras, números y guiones bajos
        nombre_limpio = ''.join(c if c.isalnum() or c == '_' else '_' for c in str(col))
        # Asegurar que empiece con letra
        if nombre_limpio and not nombre_limpio[0].isalpha():
            nombre_limpio = 'var_' + nombre_limpio
        # Evitar nombres vacíos
        if not nombre_limpio:
            nombre_limpio = f'variable_{len(mapeo_nombres)}'
        
        mapeo_nombres[col] = nombre_limpio
    
    # Renombrar columnas
    df_limpio = df_limpio.rename(columns=mapeo_nombres)
    
    return df_limpio, mapeo_nombres

# Función para realizar MANOVA
def realizar_manova(df, variables_dependientes, variable_grupo):
    """Realiza el análisis MANOVA"""
    try:
        # Limpiar nombres de columnas
        df_limpio, mapeo = limpiar_nombres_columnas(df)
        
        # Mapear nombres de variables a nombres limpios
        variables_limpias = [mapeo[var] for var in variables_dependientes]
        grupo_limpio = mapeo[variable_grupo]
        
        # Verificar que tengamos suficientes datos
        if len(df_limpio) < 10:
            return None, "Se necesitan al menos 10 observaciones para un análisis confiable"
        
        # Verificar que cada grupo tenga suficientes observaciones
        conteo_grupos = df_limpio[grupo_limpio].value_counts()
        if conteo_grupos.min() < 3:
            return None, f"Cada grupo debe tener al menos 3 observaciones. Grupo con menos datos: {conteo_grupos.min()}"
        
        # Eliminar filas con valores faltantes
        df_limpio = df_limpio.dropna(subset=variables_limpias + [grupo_limpio])
        
        if len(df_limpio) == 0:
            return None, "No quedan datos después de eliminar valores faltantes"
        
        # Crear fórmula para MANOVA
        formula = f"{' + '.join(variables_limpias)} ~ {grupo_limpio}"
        
        # Realizar MANOVA
        manova = MANOVA.from_formula(formula, data=df_limpio)
        resultado = manova.mv_test()
        
        return resultado, None
        
    except Exception as e:
        # Intentar método alternativo con scipy
        try:
            return realizar_manova_scipy(df, variables_dependientes, variable_grupo)
        except Exception as e2:
            return None, f"Error en MANOVA: {str(e)}. Error alternativo: {str(e2)}"

# Función alternativa usando scipy
def realizar_manova_scipy(df, variables_dependientes, variable_grupo):
    """Método alternativo para MANOVA usando scipy"""
    from scipy.stats import f
    
    # Preparar datos
    grupos = df[variable_grupo].unique()
    n_grupos = len(grupos)
    
    if n_grupos < 2:
        return None, "Se necesitan al menos 2 grupos para el análisis"
    
    # Calcular matrices necesarias para MANOVA
    X = df[variables_dependientes].values
    y = df[variable_grupo].values
    
    # Calcular medias por grupo
    medias_grupo = {}
    for grupo in grupos:
        mask = y == grupo
        medias_grupo[grupo] = X[mask].mean(axis=0)
    
    # Media global
    media_global = X.mean(axis=0)
    
    # Matriz de suma de cuadrados entre grupos (H)
    H = np.zeros((len(variables_dependientes), len(variables_dependientes)))
    for grupo in grupos:
        mask = y == grupo
        n_grupo = mask.sum()
        diff = medias_grupo[grupo] - media_global
        H += n_grupo * np.outer(diff, diff)
    
    # Matriz de suma de cuadrados dentro de grupos (E)
    E = np.zeros((len(variables_dependientes), len(variables_dependientes)))
    for grupo in grupos:
        mask = y == grupo
        X_grupo = X[mask]
        for i in range(len(X_grupo)):
            diff = X_grupo[i] - medias_grupo[grupo]
            E += np.outer(diff, diff)
    
    # Calcular estadísticos
    try:
        # Pillai's Trace
        eigenvals = np.linalg.eigvals(H @ np.linalg.inv(E))
        pillai_trace = np.sum(eigenvals / (1 + eigenvals))
        
        # Aproximación F para Pillai's Trace
        p = len(variables_dependientes)
        df_hyp = p * (n_grupos - 1)
        df_err = len(df) - n_grupos
        
        # Asegurar que df_err - p + 1 no sea cero o negativo
        den_df_pillai = max(1, df_err - p + 1)
        
        # Evitar división por cero para el cálculo de f_stat si pillai_trace es 1 (separación perfecta)
        if (1 - pillai_trace) == 0:
            f_stat = np.inf
        else:
            f_stat = (pillai_trace / (p * (n_grupos - 1))) * ((den_df_pillai) / (1 - pillai_trace))
        
        # Manejar casos donde f_stat podría ser muy grande o infinito
        if np.isinf(f_stat):
            p_valor = 0.0 # Altamente significativo
        elif np.isnan(f_stat):
            p_valor = np.nan # No se puede calcular
        else:
            p_valor = 1 - f.cdf(f_stat, df_hyp, den_df_pillai)
        
        # Crear DataFrame de resultados similar al de statsmodels
        resultado = pd.DataFrame({
            'Value': [pillai_trace],
            'Num DF': [df_hyp],
            'Den DF': [den_df_pillai],
            'F Value': [f_stat],
            'Pr > F': [p_valor]
        }, index=[variable_grupo])
        
        return resultado, None
        
    except Exception as e:
        return None, f"Error en cálculo alternativo: {str(e)}"

# Sidebar para configuración
st.sidebar.header("🔧 Configuración del Análisis")

# Opción para cargar datos
opcion_datos = st.sidebar.radio(
    "¿Cómo quieres cargar los datos?",
    ["📁 Subir mi archivo CSV", "📊 Usar datos de ejemplo"]
)

# Cargar datos
df = None
if opcion_datos == "📁 Subir mi archivo CSV":
    st.sidebar.subheader("📁 Carga tu archivo CSV")
    archivo_csv = st.sidebar.file_uploader(
        "Selecciona tu archivo CSV",
        type=['csv'],
        help="El archivo debe contener columnas numéricas para gastos y una columna categórica para agrupar"
    )
    
    # Opciones adicionales para la carga
    with st.sidebar.expander("⚙️ Opciones avanzadas de carga"):
        separador = st.selectbox(
            "Separador del CSV:",
            [",", ";", "\t", "|"],
            help="Carácter que separa las columnas"
        )
        
        decimal = st.selectbox(
            "Separador decimal:",
            [".", ","],
            help="Carácter usado para decimales"
        )
    
    if archivo_csv is not None:
        try:
            # Intentar diferentes codificaciones
            codificaciones = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            codificacion_exitosa = None
            
            for encoding in codificaciones:
                try:
                    # Resetear el puntero del archivo
                    archivo_csv.seek(0)
                    df = pd.read_csv(archivo_csv, encoding=encoding, sep=separador, decimal=decimal)
                    codificacion_exitosa = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # Si es otro tipo de error, intentar con la siguiente codificación
                    continue
            
            if df is not None:
                st.sidebar.success(f"✅ Archivo cargado: {len(df)} filas")
                st.sidebar.info(f"📝 Codificación detectada: {codificacion_exitosa}")
            else:
                st.sidebar.error("❌ No se pudo leer el archivo con ninguna codificación estándar")
                st.sidebar.markdown("""
                **Posibles soluciones:**
                1. Guarda el archivo como CSV UTF-8 desde Excel
                2. Usa un editor de texto para cambiar la codificación
                3. Verifica que el archivo sea realmente un CSV
                """)
                
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar el archivo: {str(e)}")
            st.sidebar.markdown("""
            **Posibles causas:**
            - Formato de archivo incorrecto
            - Archivo corrupto
            - Separadores no estándar
            """)
    else:
        st.info("👆 Por favor, sube tu archivo CSV usando el panel lateral")
        st.markdown("""
        ### 📋 Formato requerido del CSV:
        
        Tu archivo debe tener:
        - **Una columna categórica** para agrupar (ej: 'grupo', 'estrato', 'region')
        - **Varias columnas numéricas** con los gastos en servicios (ej: 'agua', 'luz', 'gas', etc.)
        
        **Ejemplo de estructura:**
        ```
        grupo,agua,luz,gas,internet,telefono
        Bajo,45.50,85.20,35.10,25.00,20.50
        Medio,78.30,145.80,65.40,55.20,42.10
        Alto,115.60,245.90,95.80,95.50,68.30
        ```
        
        ### 🔧 Problemas comunes y soluciones:
        
        **Si tienes problemas de codificación:**
        - Abre tu archivo en Excel
        - Ve a "Archivo" → "Guardar como"
        - Selecciona "CSV UTF-8 (delimitado por comas)"
        - Guarda con ese formato
        
        **Si tienes separadores diferentes:**
        - Usa las opciones avanzadas en el panel lateral
        - Prueba con punto y coma (;) si tu archivo viene de Excel en español
        
        **Si tienes decimales con coma:**
        - Cambia el separador decimal en las opciones avanzadas
        """)
        
        # Mostrar ejemplo de archivo descargable
        st.markdown("### 📥 Descargar archivo de ejemplo")
        ejemplo_data = {
            'grupo': ['Bajo', 'Bajo', 'Medio', 'Medio', 'Alto', 'Alto'],
            'agua': [45.50, 48.20, 78.30, 82.10, 115.60, 120.80],
            'luz': [85.20, 90.10, 145.80, 150.20, 245.90, 260.30],
            'gas': [35.10, 38.50, 65.40, 68.20, 95.80, 100.10],
            'internet': [25.00, 28.30, 55.20, 58.90, 95.50, 98.20],
            'telefono': [20.50, 22.10, 42.10, 45.30, 68.30, 72.40]
        }
        ejemplo_df = pd.DataFrame(ejemplo_data)
        csv_ejemplo = ejemplo_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV de ejemplo",
            data=csv_ejemplo,
            file_name="ejemplo_servicios_basicos.csv",
            mime="text/csv"
        )
else:
    df = generar_datos_ejemplo()
    st.sidebar.success("✅ Usando datos de ejemplo")

# Continuar solo si hay datos
if df is not None:
    # Validar datos
    errores = validar_datos(df)
    if errores:
        st.error("❌ Errores en los datos:")
        for error in errores:
            st.error(f"• {error}")
        st.stop()
    
    # Configuración del análisis
    st.sidebar.subheader("⚙️ Configuración del Análisis")
    
    # Seleccionar columnas
    columnas_disponibles = df.columns.tolist()
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Variable de agrupación
    if columnas_categoricas:
        variable_grupo = st.sidebar.selectbox(
            "Selecciona la variable de agrupación:",
            columnas_categoricas,
            help="Variable categórica que define los grupos a comparar"
        )
    else:
        st.sidebar.error("❌ No se encontraron columnas categóricas")
        st.stop()
    
    # Variables dependientes
    if columnas_numericas:
        variables_dependientes = st.sidebar.multiselect(
            "Selecciona las variables de gasto:",
            columnas_numericas,
            default=columnas_numericas[:5] if len(columnas_numericas) >= 5 else columnas_numericas,
            help="Variables numéricas que representan los gastos en servicios"
        )
        
        if len(variables_dependientes) < 2:
            st.sidebar.error("❌ Selecciona al menos 2 variables de gasto")
            st.stop()
    else:
        st.sidebar.error("❌ No se encontraron columnas numéricas")
        st.stop()
    
    # Otras configuraciones
    mostrar_datos = st.sidebar.checkbox("Mostrar datos crudos", value=False)
    nivel_confianza = st.sidebar.slider("Nivel de confianza", 0.90, 0.99, 0.95, 0.01)
    
    # Filtrar datos según selección
    df_analisis = df[[variable_grupo] + variables_dependientes].copy()
    
    # Eliminar filas con valores faltantes
    df_analisis = df_analisis.dropna()
    
    # Verificar que haya al menos 2 grupos
    grupos_unicos = df_analisis[variable_grupo].unique()
    if len(grupos_unicos) < 2:
        st.error("❌ Se necesitan al menos 2 grupos diferentes para el análisis")
        st.stop()
    
    # Crear columna total si no existe
    if 'total' not in df_analisis.columns:
        df_analisis['total'] = df_analisis[variables_dependientes].sum(axis=1)
    
    # Sección 1: Introducción y contexto
    st.header("🏠 Capítulo 1: El Contexto del Problema")
    st.markdown(f"""
    ### La Historia detrás de los Números

    En este análisis, exploraremos los patrones de gasto en servicios básicos entre diferentes grupos.
    Utilizaremos **{len(df_analisis)} observaciones** divididas en **{len(grupos_unicos)} grupos** para entender si existen diferencias significativas en el comportamiento de gasto.

    **Nuestra pregunta de investigación:** ¿Existen diferencias significativas en los patrones de gasto 
    en servicios básicos entre los diferentes grupos analizados?

    **Variables analizadas:**
    - **Variable de agrupación:** {variable_grupo}
    - **Variables de gasto:** {', '.join(variables_dependientes)}

    Para responder esta pregunta, utilizaremos **MANOVA (Análisis Multivariado de Varianza)**, 
    una técnica estadística que nos permite analizar múltiples variables dependientes simultáneamente.
    """)

    if mostrar_datos:
        st.subheader("📋 Datos del Análisis")
        st.dataframe(df_analisis)
        
        # Estadísticas descriptivas
        st.subheader("📊 Estadísticas Descriptivas")
        stats_desc = df_analisis.groupby(variable_grupo)[variables_dependientes].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(stats_desc)

    # Sección 2: Análisis Exploratorio
    st.header("📈 Capítulo 2: Explorando los Patrones de Gasto")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución por Grupo")
        conteo_grupos = df_analisis[variable_grupo].value_counts()
        fig_dist = px.pie(values=conteo_grupos.values, names=conteo_grupos.index, 
                          title=f"Distribución de Observaciones por {variable_grupo}")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader("Gasto Total Promedio por Grupo")
        gasto_promedio = df_analisis.groupby(variable_grupo)['total'].mean().reset_index()
        fig_bar = px.bar(gasto_promedio, x=variable_grupo, y='total', 
                         title="Gasto Total Promedio por Grupo",
                         color=variable_grupo)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Análisis por servicios individuales
    st.subheader("🔍 Análisis Detallado por Variable")

    servicio_seleccionado = st.selectbox("Selecciona una variable para analizar:", variables_dependientes)

    fig_violin = px.violin(df_analisis, x=variable_grupo, y=servicio_seleccionado, 
                           title=f"Distribución de {servicio_seleccionado}",
                           color=variable_grupo)
    st.plotly_chart(fig_violin, use_container_width=True)

    # Información de diagnóstico
    st.subheader("🔍 Diagnóstico de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de observaciones", len(df_analisis))
        st.metric("Observaciones después de limpiar", len(df_analisis.dropna()))
    
    with col2:
        st.metric("Número de grupos", len(grupos_unicos))
        min_obs_grupo = df_analisis[variable_grupo].value_counts().min()
        st.metric("Mínimo obs. por grupo", min_obs_grupo)
    
    with col3:
        st.metric("Variables analizadas", len(variables_dependientes))
        valores_faltantes = df_analisis[variables_dependientes].isnull().sum().sum()
        st.metric("Valores faltantes", valores_faltantes)
    
    # Mostrar distribución por grupo
    st.subheader("📊 Distribución de Observaciones por Grupo")
    dist_grupos = df_analisis[variable_grupo].value_counts().sort_index()
    st.dataframe(dist_grupos.to_frame("Cantidad"))
    
    # Advertencias
    if min_obs_grupo < 3:
        st.warning(f"⚠️ Algunos grupos tienen menos de 3 observaciones. Mínimo: {min_obs_grupo}")
    
    if valores_faltantes > 0:
        st.warning(f"⚠️ Hay {valores_faltantes} valores faltantes que serán eliminados del análisis")
    
    # Verificar variabilidad
    variabilidad = df_analisis[variables_dependientes].std()
    variables_sin_variabilidad = variabilidad[variabilidad == 0].index.tolist()
    if variables_sin_variabilidad:
        st.warning(f"⚠️ Variables sin variabilidad: {', '.join(variables_sin_variabilidad)}")
    
    # Mostrar primeras filas para verificar datos
    with st.expander("👀 Vista previa de datos"):
        st.dataframe(df_analisis.head(10))

    # Sección 3: Análisis MANOVA
    st.header("🎯 Capítulo 3: El Análisis MANOVA")

    st.markdown("""
    ### ¿Qué es MANOVA?

    **MANOVA (Multivariate Analysis of Variance)** es una extensión del ANOVA que nos permite:
    - Analizar múltiples variables dependientes simultáneamente
    - Controlar el error tipo I al hacer múltiples comparaciones
    - Detectar diferencias que podrían no ser evidentes en análisis univariados

    **Hipótesis de nuestro estudio:**
    - **H₀:** No hay diferencias significativas en los gastos entre grupos
    - **H₁:** Existen diferencias significativas en al menos una variable entre grupos
    """)

    # Realizar MANOVA
    with st.spinner("Realizando análisis MANOVA..."):
        resultado_manova, error_manova = realizar_manova(df_analisis, variables_dependientes, variable_grupo)

    if error_manova:
        st.error(f"❌ Error en el análisis MANOVA: {error_manova}")
        st.markdown("""
        **Posibles causas del error:**
        - Nombres de columnas con caracteres especiales
        - Muy pocas observaciones por grupo (mínimo 3 por grupo)
        - Variables con poca variabilidad
        - Problemas de multicolinealidad
        - Valores faltantes en los datos
        
        **Recomendaciones:**
        1. Verifica que cada grupo tenga al menos 3 observaciones
        2. Revisa que no haya valores faltantes
        3. Asegúrate de que las variables numéricas tengan variabilidad
        4. Simplifica los nombres de columnas (sin espacios ni caracteres especiales)
        """)
    else:
        st.subheader("📊 Resultados del Análisis MANOVA")

        # Mostrar resultados
        st.markdown("### Estadísticos de Prueba")
        # Acceder a los DataFrames de los resultados (Pillai's Trace, Wilks' Lambda, etc.) desde el diccionario de resultados.
        # El método 'mv_test()' usualmente retorna un diccionario con claves como 'Target', 'Intercept', etc.
        # Estamos interesados en la tabla para la variable independiente (variable_grupo).
        
        # Esto obtendrá el DataFrame para tu variable independiente
        if isinstance(resultado_manova, dict) and variable_grupo in resultado_manova:
            resultado_df = resultado_manova[variable_grupo].round(4)
        elif isinstance(resultado_manova, pd.DataFrame): # Para el caso de fallback de scipy
            resultado_df = resultado_manova.round(4)
        else:
            st.warning("No se pudo formatear los resultados del MANOVA. Mostrando el objeto completo.")
            st.write(resultado_manova)
            resultado_df = pd.DataFrame() # Crear un DataFrame vacío para evitar errores posteriores

        if not resultado_df.empty:
            st.dataframe(resultado_df)

            # Interpretación de resultados
            st.markdown("### 🔍 Interpretación de los Resultados")

            # Obtener el p-valor (usando diferentes métodos según el resultado)
            # Priorizar 'Pr > F' y manejar casos donde la estructura podría ser diferente
            p_valor_pillai = np.nan
            if 'Pr > F' in resultado_df.columns:
                p_valor_pillai = resultado_df.loc[resultado_df.index[0], "Pr > F"]
            elif 'P-value' in resultado_df.columns: # Para posibles cambios futuros u otras librerías
                p_valor_pillai = resultado_df.loc[resultado_df.index[0], "P-value"]
            elif resultado_df.shape[1] > 0: # Intentar obtener la última columna si no se encuentra 'Pr > F'
                p_valor_pillai = resultado_df.iloc[0, -1]
            
            # Asegurarse de que sea un float para la comparación
            if isinstance(p_valor_pillai, (pd.Series, pd.DataFrame)):
                p_valor_pillai = p_valor_pillai.iloc[0]
            try:
                p_valor_pillai = float(p_valor_pillai)
            except (ValueError, TypeError):
                p_valor_pillai = np.nan # Si la conversión falla, establecer a NaN

            if not np.isnan(p_valor_pillai) and p_valor_pillai < (1 - nivel_confianza):
                st.success(f"""
                **✅ RESULTADO SIGNIFICATIVO** (p = {p_valor_pillai:.4f})
                
                Con un nivel de confianza del {nivel_confianza*100}%, podemos **rechazar la hipótesis nula**.
                
                **Conclusión:** Existen diferencias estadísticamente significativas en los patrones de gasto 
                entre los diferentes grupos analizados.
                """)
            elif not np.isnan(p_valor_pillai):
                st.warning(f"""
                **❌ RESULTADO NO SIGNIFICATIVO** (p = {p_valor_pillai:.4f})
                
                Con un nivel de confianza del {nivel_confianza*100}%, **no podemos rechazar la hipótesis nula**.
                
                **Conclusión:** No hay evidencia suficiente para afirmar que existen diferencias significativas 
                en los gastos entre los grupos analizados.
                """)
            else:
                st.warning("No se pudo obtener un p-valor para la interpretación. Revise los resultados completos.")

    # Sección 4: Análisis Post-Hoc
    st.header("🔬 Capítulo 4: Análisis Detallado por Variable")

    st.markdown("""
    Aunque MANOVA nos da una respuesta global, es importante entender **qué variables específicas** muestran las mayores diferencias entre grupos.
    """)

    # Análisis univariado para cada variable
    st.subheader("Análisis ANOVA Individual por Variable")

    resultados_anova = []
    for variable in variables_dependientes:
        try:
            # ANOVA para cada variable
            grupos_variable = [df_analisis[df_analisis[variable_grupo] == grupo][variable].dropna() 
                               for grupo in grupos_unicos]
            # Filtrar grupos vacíos
            grupos_variable = [grupo for grupo in grupos_variable if len(grupo) > 0]
            
            # Asegurarse de que haya al menos 2 grupos no vacíos y más de 1 observación por grupo para un ANOVA significativo
            if len(grupos_variable) >= 2 and all(len(g) > 1 for g in grupos_variable): 
                f_stat, p_val = stats.f_oneway(*grupos_variable)
                
                resultados_anova.append({
                    'Variable': variable,
                    'F-estadístico': round(f_stat, 4),
                    'p-valor': round(p_val, 4),
                    'Significativo': 'Sí' if p_val < (1 - nivel_confianza) else 'No'
                })
            else:
                resultados_anova.append({
                    'Variable': variable,
                    'F-estadístico': 'N/A',
                    'p-valor': 'N/A',
                    'Significativo': 'Datos insuficientes'
                })
        except Exception as e:
            resultados_anova.append({
                'Variable': variable,
                'F-estadístico': 'Error',
                'p-valor': 'Error',
                'Significativo': 'Error'
            })

    df_anova = pd.DataFrame(resultados_anova)
    st.dataframe(df_anova)

    # Visualización de diferencias entre grupos
    st.subheader("📊 Comparación Visual entre Grupos")

    # Crear gráfico de barras agrupadas
    n_vars = len(variables_dependientes)
    cols = 3
    rows = (n_vars + cols - 1) // cols

    fig_comparacion = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=variables_dependientes,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    for i, variable in enumerate(variables_dependientes):
        row = i // cols + 1
        col = i % cols + 1
        
        medias = df_analisis.groupby(variable_grupo)[variable].mean()
        
        fig_comparacion.add_trace(
            go.Bar(x=medias.index, y=medias.values, 
                   name=variable,
                   showlegend=False),
            row=row, col=col
        )

    fig_comparacion.update_layout(height=200*rows, title_text="Gasto Promedio por Variable y Grupo")
    st.plotly_chart(fig_comparacion, use_container_width=True)

    # Sección 5: Conclusiones
    st.header("🎯 Capítulo 5: Conclusiones y Recomendaciones")

    st.markdown(f"""
    ### 📋 Resumen Ejecutivo

    Basado en nuestro análisis MANOVA de **{len(df_analisis)} observaciones**:

    **Principales Hallazgos:**
    """)

    # Calcular estadísticas descriptivas por grupo
    cols_metricas = st.columns(len(grupos_unicos))
    
    for i, grupo in enumerate(grupos_unicos):
        datos_grupo = df_analisis[df_analisis[variable_grupo] == grupo]
        with cols_metricas[i]:
            st.metric(f"Grupo: {grupo}", len(datos_grupo))
            st.metric(f"Gasto Promedio", f"${datos_grupo['total'].mean():.2f}")
            st.metric(f"Desviación Estándar", f"${datos_grupo['total'].std():.2f}")

    # Recomendaciones personalizadas
    st.markdown("""
    ### 💡 Recomendaciones

    **Basado en el análisis de tus datos:**

    1. **Si encontraste diferencias significativas:**
        - Investiga qué factores causan estas diferencias
        - Considera estrategias diferenciadas por grupo
        - Desarrolla políticas segmentadas para cada grupo

    2. **Si no encontraste diferencias significativas:**
        - Los grupos muestran patrones similares de gasto
        - Puedes aplicar estrategias uniformes
        - Considera otros factores de segmentación

    3. **Para análisis futuros:**
        - Incluye más variables explicativas
        - Considera análisis de series temporales
        - Evalúa factores externos que puedan influir
    """)

    # Exportar resultados
    st.subheader("📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Descargar Estadísticas Descriptivas"):
            stats_export = df_analisis.groupby(variable_grupo)[variables_dependientes].agg(['count', 'mean', 'std']).round(2)
            csv = stats_export.to_csv()
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="estadisticas_descriptivas.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔬 Descargar Resultados ANOVA"):
            csv_anova = df_anova.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_anova,
                file_name="resultados_anova.csv",
                mime="text/csv"
            )

    # Sección técnica
    with st.expander("🔧 Detalles Técnicos del Análisis"):
        st.markdown(f"""
        **Metodología Empleada:**
        - **Técnica:** MANOVA (Multivariate Analysis of Variance)
        - **Variables Dependientes:** {', '.join(variables_dependientes)}
        - **Variable Independiente:** {variable_grupo}
        - **Número de grupos:** {len(grupos_unicos)}
        - **Tamaño de muestra:** {len(df_analisis)}
        - **Estadístico Principal:** Pillai's Trace (más robusto ante violaciones de supuestos)
        - **Software:** Python con librerías statsmodels, scipy, plotly
        
        **Supuestos del MANOVA:**
        - Normalidad multivariada
        - Homogeneidad de matrices de covarianza
        - Independencia de observaciones
        - Ausencia de outliers extremos
        
        **Grupos analizados:** {', '.join(grupos_unicos)}
        """)

# Footer
st.markdown("---")
st.markdown("*Desarrollado con  usando Python y Streamlit para análisis estadístico*")