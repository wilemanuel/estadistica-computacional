import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io

st.set_page_config(page_title="Análisis ENAHO 2022", layout="wide")
st.title("📊 Análisis Visual de Datos ENAHO")

# Subida de archivo
archivo = st.file_uploader("📁 Sube tu archivo CSV", type=["csv"])

df = None # Inicializar df fuera del bloque if

if archivo is not None:
    try:
        # Intentar cargar con codificaciones comunes
        codificaciones = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        codificacion_exitosa = None
        for encoding in codificaciones:
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding=encoding, low_memory=False)
                codificacion_exitosa = encoding
                st.success(f"✅ Datos cargados correctamente con codificación: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error al leer con {encoding}: {e}")
                df = None
        
        if df is None:
            st.error("❌ No se pudo cargar el archivo CSV con ninguna de las codificaciones intentadas. Por favor, verifica el formato y la codificación de tu archivo.")
            st.stop()

        st.dataframe(df.head())

        # --- Limpiar los nombres de las columnas para mayor robustez ---
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '_').str.replace('-', '_')
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.map('_'.join)
            st.info("Detectado y simplificado un índice de columnas multi-nivel.")
        
        if original_columns != df.columns.tolist():
            st.info("Se han limpiado los nombres de las columnas. Los nuevos nombres son: " + ", ".join(df.columns.tolist()))

        # Selección de todas las columnas numéricas disponibles
        todas_columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
        
        # --- 1. Matriz de correlación ---
        st.subheader("1️⃣ Matriz de correlación de Variables Numéricas Relevantes")
        st.markdown("Selecciona las variables numéricas que consideres más relevantes para analizar su correlación.")

        # Nuevo: Selector múltiple para elegir las columnas a correlacionar
        selected_numeric_columns = st.multiselect(
            "Elige las columnas numéricas para la matriz de correlación:",
            todas_columnas_numericas,
            default=todas_columnas_numericas[:5] if len(todas_columnas_numericas) > 0 else [] # Selecciona las primeras 5 por defecto si existen
        )

        if len(selected_numeric_columns) >= 2:
            corr = df[selected_numeric_columns].corr()
            
            fig_corr = px.imshow(
                corr,
                text_auto=".2f", # Formato a 2 decimales
                aspect="auto",
                color_continuous_scale='RdBu_r', # Escala de color Rojo-Azul invertida
                title="Mapa de Correlación entre Variables Numéricas Seleccionadas"
            )
            
            # Ajustar el tamaño del texto dentro de las celdas y las etiquetas de los ejes
            # El tamaño de la fuente se hará más grande si hay menos columnas
            font_size_labels = 12 if len(selected_numeric_columns) < 15 else 8
            font_size_text = 12 if len(selected_numeric_columns) < 10 else 8

            fig_corr.update_traces(textfont_size=font_size_text) 
            fig_corr.update_xaxes(tickangle=-45, tickfont_size=font_size_labels)
            fig_corr.update_yaxes(tickfont_size=font_size_labels)

            # Ajustar layout para mejor visualización
            fig_corr.update_layout(
                height=max(500, len(selected_numeric_columns) * 40), # Altura dinámica
                width=max(500, len(selected_numeric_columns) * 40), # Ancho dinámico
                title_x=0.5, 
                title_font_size=20 
            ) 

            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            **Interpretación de la Matriz de Correlación:**
            - **Valores cercanos a 1:** Indican una fuerte correlación positiva (cuando una variable aumenta, la otra también tiende a aumentar).
            - **Valores cercanos a -1:** Indican una fuerte correlación negativa (cuando una variable aumenta, la otra tiende a disminuir).
            - **Valores cercanos a 0:** Indican poca o ninguna correlación lineal entre las variables.
            """)

        else:
            st.warning("Por favor, selecciona al menos 2 columnas numéricas para generar la matriz de correlación.")

        # --- Validación y selección de la columna de resultado para los siguientes gráficos ---
        available_columns = df.columns.tolist()
        
        # Filtrar solo columnas numéricas para el resultado por defecto, si es posible
        opciones_resultado = [col for col in available_columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if not opciones_resultado:
            st.error("No se encontraron columnas numéricas para usar como 'Resultado de la entrevista'. Algunos gráficos no podrán generarse.")
            col_resultado_comun = None
        else:
            col_resultado_comun = st.selectbox(
                "Selecciona la columna de **Resultado de la entrevista** (para gráficos 2, 3 y 4)", 
                opciones_resultado, # Limitar a columnas numéricas
                key="resultado_select_comun"
            )
            # Asegurarse de que la columna seleccionada sigue siendo numérica (por si el usuario elige otra cosa en un set distinto)
            if not pd.api.types.is_numeric_dtype(df[col_resultado_comun]):
                st.warning("La columna de Resultado seleccionada no es numérica. Por favor, selecciona una columna numérica.")
                col_resultado_comun = None # Invalidar si no es numérica

        if col_resultado_comun: # Solo proceder si hay una columna de resultado numérica válida
            # --- 2. Gráfico de línea: Estrato vs Resultado ---
            st.subheader("2️⃣ ¿Influye el estrato social en el resultado?")
            
            col_estrato = st.selectbox("Selecciona la columna de Estrato Social", available_columns, key="estrato_select")

            st.info(f"DEBUG: Columna 'Estrato Social' seleccionada: '{col_estrato}'")
            st.info(f"DEBUG: Tipo de dato de '{col_estrato}': {df[col_estrato].dtype}")
            st.info(f"DEBUG: Columna 'Resultado' seleccionada: '{col_resultado_comun}'")
            st.info(f"DEBUG: Tipo de dato de '{col_resultado_comun}': {df[col_resultado_comun].dtype}")

            if col_estrato not in df.columns:
                st.error("La columna de estrato seleccionada no es válida.")
            else:
                if not pd.api.types.is_numeric_dtype(df[col_estrato]) and not pd.api.types.is_categorical_dtype(df[col_estrato]):
                    try:
                        df[col_estrato] = df[col_estrato].astype('category')
                        st.info(f"DEBUG: Columna '{col_estrato}' convertida a tipo 'category' para agrupación.")
                    except Exception as e:
                        st.error(f"Error al intentar convertir '{col_estrato}' a categoría: {e}. Esto podría causar problemas en la agrupación.")

                df_line = df[[col_estrato, col_resultado_comun]].dropna()
                
                if df_line.empty:
                    st.warning(f"No hay datos disponibles para '{col_estrato}' y '{col_resultado_comun}' después de eliminar valores faltantes.")
                else:
                    if df_line[col_estrato].nunique() < 2:
                        st.warning(f"La columna '{col_estrato}' tiene muy pocos valores únicos ({df_line[col_estrato].nunique()}) para un análisis significativo. Necesitas al menos 2 grupos.")
                    else:
                        try:
                            st.info(f"DEBUG: Tipo de df_line['{col_estrato}'] antes de groupby: {df_line[col_estrato].dtype}")
                            st.info(f"DEBUG: Shape de df_line['{col_estrato}'] antes de groupby: {df_line[col_estrato].shape}")

                            df_grouped = df_line.groupby(col_estrato)[col_resultado_comun].mean().reset_index()
                            
                            fig2 = px.line(df_grouped, x=col_estrato, y=col_resultado_comun,
                                           markers=True, title=f"Tendencia del promedio de {col_resultado_comun} según {col_estrato}")
                            st.plotly_chart(fig2, use_container_width=True)
                            st.markdown(f"""
                            **Interpretación:** Este gráfico de línea muestra cómo el valor promedio de **'{col_resultado_comun}'**
                            varía a través de las diferentes categorías de **'{col_estrato}'**.
                            Puedes observar tendencias o patrones en los resultados a medida que cambias de estrato.
                            """)
                        except Exception as e:
                            st.error(f"Error al generar el gráfico de línea: {e}. Esto podría deberse a datos inesperados en la columna de agrupación, o si la columna '{col_estrato}' tiene una estructura compleja (ej. MultiIndex).")
                            st.info(f"Asegúrate de que la columna '{col_estrato}' sea un tipo de dato simple y 1-dimensional (categórico, texto, o numérico discreto).")


            # --- 3. Gráfico de caja: Región vs Resultado ---
            st.subheader("3️⃣ ¿Hay diferencias por región?")
            col_region = st.selectbox("Selecciona la columna de Región", available_columns, key="region_select")

            if col_region not in df.columns:
                st.error("La columna de región seleccionada no es válida.")
            else:
                if not pd.api.types.is_numeric_dtype(df[col_region]) and not pd.api.types.is_categorical_dtype(df[col_region]):
                    try:
                        df[col_region] = df[col_region].astype('category')
                        st.info(f"DEBUG: Columna '{col_region}' convertida a tipo 'category' para agrupación.")
                    except Exception as e:
                        st.error(f"Error al intentar convertir '{col_region}' a categoría: {e}. Esto podría causar problemas en la agrupación.")

                df_box = df[[col_region, col_resultado_comun]].dropna()
                
                if df_box.empty:
                    st.warning(f"No hay datos disponibles para '{col_region}' y '{col_resultado_comun}' después de eliminar valores faltantes.")
                else:
                    if df_box[col_region].nunique() < 2:
                        st.warning(f"La columna '{col_region}' tiene muy pocos valores únicos ({df_box[col_region].nunique()}) para un análisis significativo. Necesitas al menos 2 grupos.")
                    else:
                        fig3 = px.box(df_box, x=col_region, y=col_resultado_comun,
                                      title=f"Distribución de {col_resultado_comun} por {col_region}")
                        st.plotly_chart(fig3, use_container_width=True)
                        st.markdown(f"""
                        **Interpretación:** Los diagramas de caja muestran la distribución de **'{col_resultado_comun}'**
                        para cada categoría de **'{col_region}'**. Puedes identificar:
                        - **Mediana:** La línea central de la caja.
                        - **Rango intercuartílico (IQR):** La altura de la caja (donde se concentra el 50% central de los datos).
                        - **Bigotes:** La dispersión de los datos fuera del IQR.
                        - **Puntos (Outliers):** Valores atípicos que están significativamente fuera del rango.
                        Esto te ayuda a ver rápidamente si hay diferencias en la tendencia central y la dispersión entre regiones.
                        """)

            # --- 4. Visualización Combinada: Mapa de Calor de Medias ---
            st.subheader("4️⃣ Mapa de Calor de Medias por Dos Categorías")
            st.markdown("Este gráfico muestra cómo el promedio de una variable numérica se distribuye entre las combinaciones de dos variables categóricas (o discretas).")

            col_heatmap_x = st.selectbox("Selecciona la columna para el Eje X (Categoría 1)", available_columns, key="heatmap_x")
            col_heatmap_y = st.selectbox("Selecciona la columna para el Eje Y (Categoría 2)", available_columns, key="heatmap_y")
            col_heatmap_value = st.selectbox("Selecciona la columna numérica para el valor (Ej. Resultado, Ingreso)", opciones_resultado, key="heatmap_value") 

            if col_heatmap_x and col_heatmap_y and col_heatmap_value:
                if col_heatmap_x not in df.columns or col_heatmap_y not in df.columns or col_heatmap_value not in df.columns:
                    st.error("Una de las columnas seleccionadas para el mapa de calor no es válida.")
                elif not pd.api.types.is_numeric_dtype(df[col_heatmap_value]):
                    st.warning("La columna de valor para el mapa de calor debe ser numérica.")
                else:
                    # Convertir a categoría si no lo son ya, para asegurar un comportamiento de agrupación predecible
                    temp_df = df[[col_heatmap_x, col_heatmap_y, col_heatmap_value]].copy()
                    if not pd.api.types.is_numeric_dtype(temp_df[col_heatmap_x]) and not pd.api.types.is_categorical_dtype(temp_df[col_heatmap_x]):
                        temp_df[col_heatmap_x] = temp_df[col_heatmap_x].astype('category')
                    if not pd.api.types.is_numeric_dtype(temp_df[col_heatmap_y]) and not pd.api.types.is_categorical_dtype(temp_df[col_heatmap_y]):
                        temp_df[col_heatmap_y] = temp_df[col_heatmap_y].astype('category')
                    
                    pivot_table = temp_df.pivot_table(index=col_heatmap_y, columns=col_heatmap_x, values=col_heatmap_value, aggfunc='mean')
                    
                    if not pivot_table.empty:
                        fig_heatmap_custom = px.imshow(
                            pivot_table,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title=f"Promedio de {col_heatmap_value} por {col_heatmap_y} y {col_heatmap_x}"
                        )
                        fig_heatmap_custom.update_xaxes(side="top")
                        fig_heatmap_custom.update_layout(height=600, width=800)
                        st.plotly_chart(fig_heatmap_custom, use_container_width=True)
                        st.markdown(f"""
                        **Interpretación:** Este mapa de calor muestra el promedio de **'{col_heatmap_value}'**
                        para cada combinación única de **'{col_heatmap_x}'** y **'{col_heatmap_y}'**.
                        Los colores más intensos (o más claros, dependiendo de la escala) indican promedios más altos (o bajos).
                        Es útil para identificar rápidamente combinaciones de categorías con resultados particulares.
                        """)
                    else:
                        st.warning("No hay suficientes datos para generar el mapa de calor con las columnas seleccionadas después de la agrupación.")
            else:
                st.info("Selecciona tres columnas (dos categóricas/discretas y una numérica) para generar el mapa de calor de medias.")


            st.markdown("✅ **Interpretación General**: Puedes ajustar los campos para personalizar el análisis según tu CSV.")
        
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar los datos o generar gráficos: {e}")
        st.info("Asegúrate de que tu archivo CSV tenga el formato esperado y que las columnas seleccionadas sean apropiadas para el tipo de gráfico.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")