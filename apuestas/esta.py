import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn 
from io import StringIO

# Configurar la página
st.set_page_config(
    page_title="📊 Analizador de CSV - Estadísticas Básicas",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Analizador de CSV - Estadísticas Básicas")
st.markdown("### ¡Sube tu archivo CSV y descubre qué dicen tus datos!")

# Sidebar con información
st.sidebar.markdown("## 📋 Guía Rápida")
st.sidebar.markdown("""
**¿Qué puedes hacer aquí?**
- Subir un archivo CSV
- Ver estadísticas básicas
- Entender qué significan los números
- Ver gráficos fáciles de entender
""")

# Función para explicar conceptos estadísticos
def explicar_concepto(concepto):
    explicaciones = {
        "promedio": "📈 **Promedio (Media)**: Es como dividir una pizza entre todos. Si tienes 10 pizzas y 5 personas, cada uno come 2 pizzas en promedio.",
        "mediana": "📊 **Mediana**: Es el valor del medio cuando ordenas todos los números de menor a mayor. Como el estudiante que queda en el medio en una fila ordenada por altura.",
        "moda": "🎯 **Moda**: Es el valor que más se repite. Como el color favorito que eligió la mayoría de personas en una encuesta.",
        "desviacion": "📏 **Desviación Estándar**: Nos dice qué tan dispersos están los datos. Si es pequeña, los datos están muy juntos. Si es grande, están muy separados.",
        "minimo": "⬇️ **Mínimo**: El valor más pequeño en tus datos.",
        "maximo": "⬆️ **Máximo**: El valor más grande en tus datos.",
        "rango": "📐 **Rango**: La diferencia entre el valor más grande y el más pequeño.",
        "cuartiles": "🎯 **Cuartiles**: Dividen tus datos en 4 partes iguales. Como dividir a todos los estudiantes en 4 grupos del mismo tamaño según sus calificaciones."
    }
    return explicaciones.get(concepto, "")

# Subir archivo
uploaded_file = st.file_uploader(
    "🔗 Sube tu archivo CSV aquí", 
    type=['csv'],
    help="Arrastra y suelta tu archivo CSV o haz clic para seleccionarlo"
)

if uploaded_file is not None:
    try:
        # Función para detectar la codificación del archivo
        def leer_csv_con_codificacion(archivo):
            # Lista de codificaciones comunes
            codificaciones = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            for codificacion in codificaciones:
                try:
                    # Resetear el puntero del archivo
                    archivo.seek(0)
                    # Intentar leer con la codificación actual
                    df = pd.read_csv(archivo, encoding=codificacion)
                    st.success(f"✅ Archivo leído correctamente con codificación: {codificacion}")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # Si hay otro error, intentar con la siguiente codificación
                    continue
            
            # Si ninguna codificación funcionó, intentar con 'errors=ignore'
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding='utf-8', errors='ignore')
                st.warning("⚠️ Se leyó el archivo ignorando algunos caracteres especiales")
                return df
            except:
                raise Exception("No se pudo leer el archivo con ninguna codificación común")
        
        # Leer el archivo CSV con detección automática de codificación
        df = leer_csv_con_codificacion(uploaded_file)
        
        # Mostrar información básica del archivo
        st.success(f"✅ ¡Archivo cargado exitosamente!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total de filas", len(df))
        with col2:
            st.metric("📊 Total de columnas", len(df.columns))
        with col3:
            st.metric("🔢 Columnas numéricas", len(df.select_dtypes(include=[np.number]).columns))
        
        # Mostrar vista previa de los datos
        st.subheader("👀 Vista previa de tus datos")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Obtener solo columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) > 0:
            st.subheader("📊 Estadísticas Básicas")
            
            # Selector de columna
            columna_seleccionada = st.selectbox(
                "🎯 Selecciona una columna para analizar:",
                columnas_numericas,
                help="Elige la columna de la que quieres ver las estadísticas"
            )
            
            if columna_seleccionada:
                datos_columna = df[columna_seleccionada].dropna()
                
                # Calcular estadísticas
                promedio = datos_columna.mean()
                mediana = datos_columna.median()
                moda = datos_columna.mode().iloc[0] if not datos_columna.mode().empty else "No hay moda"
                desviacion = datos_columna.std()
                minimo = datos_columna.min()
                maximo = datos_columna.max()
                rango = maximo - minimo
                q1 = datos_columna.quantile(0.25)
                q3 = datos_columna.quantile(0.75)
                
                # Mostrar estadísticas en tarjetas
                st.markdown(f"### 📈 Estadísticas de la columna: **{columna_seleccionada}**")
                
                # Fila 1 de métricas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 Promedio", f"{promedio:.2f}")
                    st.info(explicar_concepto("promedio"))
                    
                with col2:
                    st.metric("📊 Mediana", f"{mediana:.2f}")
                    st.info(explicar_concepto("mediana"))
                    
                with col3:
                    st.metric("⬇️ Mínimo", f"{minimo:.2f}")
                    st.info(explicar_concepto("minimo"))
                    
                with col4:
                    st.metric("⬆️ Máximo", f"{maximo:.2f}")
                    st.info(explicar_concepto("maximo"))
                
                # Fila 2 de métricas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📏 Desviación Estándar", f"{desviacion:.2f}")
                    st.info(explicar_concepto("desviacion"))
                    
                with col2:
                    st.metric("🎯 Moda", f"{moda}")
                    st.info(explicar_concepto("moda"))
                    
                with col3:
                    st.metric("📐 Rango", f"{rango:.2f}")
                    st.info(explicar_concepto("rango"))
                    
                with col4:
                    st.metric("🔢 Datos válidos", len(datos_columna))
                    st.info("Cantidad de datos que no están vacíos")
                
                # Interpretación automática
                st.subheader("🤔 ¿Qué significa esto?")
                
                interpretacion = []
                
                if abs(promedio - mediana) / promedio < 0.1:
                    interpretacion.append("✅ **Datos balanceados**: Tu promedio y mediana son muy similares, esto significa que tus datos están bien distribuidos.")
                else:
                    interpretacion.append("⚠️ **Datos desbalanceados**: Tu promedio y mediana son diferentes, esto puede indicar que tienes valores muy altos o muy bajos que afectan el promedio.")
                
                cv = (desviacion / promedio) * 100 if promedio != 0 else 0
                if cv < 15:
                    interpretacion.append(f"📊 **Datos consistentes**: Tus datos varían poco (variación del {cv:.1f}%), están bastante agrupados.")
                elif cv < 30:
                    interpretacion.append(f"📈 **Datos moderadamente variables**: Tus datos tienen variación media ({cv:.1f}%).")
                else:
                    interpretacion.append(f"📉 **Datos muy variables**: Tus datos están muy dispersos (variación del {cv:.1f}%).")
                
                for interp in interpretacion:
                    st.markdown(interp)
                
                # Gráficos
                st.subheader("📊 Visualizaciones")
                
                tab1, tab2, tab3 = st.tabs(["📊 Histograma", "📈 Gráfico de Caja", "📉 Estadísticas Resumidas"])
                
                with tab1:
                    st.markdown("**¿Qué es un histograma?** 📊")
                    st.markdown("Es como un gráfico de barras que muestra cuántos datos tienes en cada rango de valores. Te ayuda a ver la forma de tus datos.")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(datos_columna, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                    ax.set_xlabel(columna_seleccionada)
                    ax.set_ylabel('Frecuencia (¿Cuántas veces aparece?)')
                    ax.set_title(f'Distribución de {columna_seleccionada}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with tab2:
                    st.markdown("**¿Qué es un gráfico de caja?** 📈")
                    st.markdown("Es como un resumen visual de tus datos. La caja muestra dónde está la mayoría de tus datos, y las líneas muestran los valores extremos.")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.boxplot(datos_columna, vert=False)
                    ax.set_xlabel(columna_seleccionada)
                    ax.set_title(f'Resumen visual de {columna_seleccionada}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with tab3:
                    st.markdown("**📋 Resumen completo de estadísticas**")
                    
                    # Crear tabla de resumen
                    resumen = pd.DataFrame({
                        'Estadística': ['Promedio', 'Mediana', 'Moda', 'Desviación Estándar', 'Mínimo', 'Máximo', 'Rango', 'Cuartil 1 (25%)', 'Cuartil 3 (75%)'],
                        'Valor': [f"{promedio:.2f}", f"{mediana:.2f}", str(moda), f"{desviacion:.2f}", f"{minimo:.2f}", f"{maximo:.2f}", f"{rango:.2f}", f"{q1:.2f}", f"{q3:.2f}"],
                        'Significado': [
                            'Valor típico de tus datos',
                            'Valor del medio cuando ordenas los datos',
                            'Valor que más se repite',
                            'Qué tan dispersos están los datos',
                            'Valor más pequeño',
                            'Valor más grande',
                            'Diferencia entre máximo y mínimo',
                            '25% de datos están por debajo de este valor',
                            '75% de datos están por debajo de este valor'
                        ]
                    })
                    
                    st.dataframe(resumen, use_container_width=True)
                
                # Análisis de todas las columnas numéricas
                if len(columnas_numericas) > 1:
                    st.subheader("🔍 Resumen de todas las columnas numéricas")
                    
                    # Descripción estadística de todas las columnas
                    descripcion = df[columnas_numericas].describe()
                    st.dataframe(descripcion, use_container_width=True)
                    
                    st.markdown("**¿Qué significa cada fila?**")
                    st.markdown("""
                    - **count**: Cuántos datos válidos tienes en cada columna
                    - **mean**: El promedio de cada columna
                    - **std**: La desviación estándar (dispersión) de cada columna
                    - **min**: El valor mínimo de cada columna
                    - **25%**: El 25% de los datos están por debajo de este valor
                    - **50%**: La mediana (valor del medio)
                    - **75%**: El 75% de los datos están por debajo de este valor
                    - **max**: El valor máximo de cada columna
                    """)
        else:
            st.warning("😔 No se encontraron columnas numéricas en tu archivo CSV. Las estadísticas solo se pueden calcular para datos numéricos.")
            st.markdown("**¿Qué puedes hacer?**")
            st.markdown("- Verifica que tu archivo tenga columnas con números")
            st.markdown("- Asegúrate de que los números no estén escritos como texto")
            
    except Exception as e:
        st.error(f"😞 Hubo un error al procesar tu archivo: {str(e)}")
        st.markdown("**Posibles soluciones:**")
        st.markdown("- **Problema de codificación**: Abre tu archivo CSV en un editor de texto y guárdalo como UTF-8")
        st.markdown("- **Archivo dañado**: Verifica que tu archivo CSV no esté corrupto")
        st.markdown("- **Formato incorrecto**: Asegúrate de que sea un archivo CSV válido")
        st.markdown("- **Caracteres especiales**: Evita usar caracteres como ñ, á, é, í, ó, ú en los nombres de columnas")
        
        # Mostrar información adicional para debug
        st.markdown("### 🔧 Información técnica:")
        st.code(f"Error específico: {str(e)}", language='text')
        
        # Sugerencias específicas para codificación
        st.markdown("### 💡 Si tienes problemas de codificación:")
        st.markdown("""
        **Opción 1 - En Excel:**
        1. Abre tu archivo en Excel
        2. Ve a 'Archivo' → 'Guardar como'
        3. Selecciona 'CSV UTF-8 (delimitado por comas)'
        4. Guarda el archivo
        
        **Opción 2 - En Google Sheets:**
        1. Sube tu archivo a Google Sheets
        2. Ve a 'Archivo' → 'Descargar' → 'Valores separados por comas (.csv)'
        3. Usa el archivo descargado
        """)
        
        st.markdown("### 🆘 ¿Sigues teniendo problemas?")
        st.markdown("Envíame las primeras líneas de tu archivo CSV y te ayudo a solucionarlo.")

else:
    # Mostrar información de ayuda cuando no hay archivo
    st.info("👆 Sube un archivo CSV para comenzar el análisis")
    
    st.markdown("### 📚 ¿Qué son las estadísticas básicas?")
    st.markdown("""
    Las estadísticas básicas son como un resumen de tus datos que te ayuda a entender:
    
    - **📈 ¿Cuál es el valor típico?** (Promedio)
    - **📊 ¿Cuál es el valor del medio?** (Mediana)
    - **🎯 ¿Cuál es el valor más común?** (Moda)
    - **📏 ¿Qué tan dispersos están los datos?** (Desviación estándar)
    - **⬇️⬆️ ¿Cuáles son los valores extremos?** (Mínimo y máximo)
    """)
    
    st.markdown("### 📋 Ejemplo de archivo CSV")
    st.markdown("Tu archivo CSV debe verse algo así:")
    
    ejemplo_csv = """Nombre,Edad,Salario,Experiencia
Juan,25,45000,2
María,30,55000,5
Pedro,35,65000,8
Ana,28,50000,3"""
    
    st.code(ejemplo_csv, language='csv')
    
    st.markdown("**💡 Consejos para tu archivo CSV:**")
    st.markdown("- La primera fila debe contener los nombres de las columnas")
    st.markdown("- Los datos deben estar separados por comas")
    st.markdown("- Los números no deben contener símbolos como $ o %")
    st.markdown("- Guarda el archivo con extensión .csv")

# Footer
st.markdown("---")
st.markdown("💡 **Consejo**: Las estadísticas te ayudan a tomar mejores decisiones basadas en tus datos. ¡No necesitas ser un experto para entender lo que significan!")