import streamlit as st
import random
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Imports opcionales
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class CandyGameModel:
    def __init__(self, 
                 candy_types=['limon', 'huevo', 'pera'],
                 num_people=10,
                 candies_per_person=2,
                 candies_per_lollipop=6,
                 candies_from_selling=6,
                 extra_candies_from_making=2):
        
        self.candy_types = candy_types
        self.num_people = num_people
        self.candies_per_person = candies_per_person
        self.candies_per_lollipop = candies_per_lollipop
        self.candies_from_selling = candies_from_selling
        self.extra_candies_from_making = extra_candies_from_making
        
        # Calculamos cuántos de cada tipo necesitamos para hacer un chupetín
        self.candies_per_type_for_lollipop = max(1, candies_per_lollipop // len(candy_types))
        
    def can_make_lollipop(self, inventory):
        """Verifica si se puede hacer un chupetín con el inventario actual"""
        for candy_type in self.candy_types:
            if inventory[candy_type] < self.candies_per_type_for_lollipop:
                return False
        return True
    
    def make_lollipop(self, inventory, steps):
        """Hace un chupetín y actualiza el inventario"""
        # Usar los caramelos necesarios
        for candy_type in self.candy_types:
            inventory[candy_type] -= self.candies_per_type_for_lollipop
        
        # Agregar caramelos extra estratégicos
        missing = self.get_missing_candies(inventory)
        extra = []
        
        for candy, amount in missing.items():
            extra.extend([candy] * min(amount, self.extra_candies_from_making - len(extra)))
        
        while len(extra) < self.extra_candies_from_making:
            extra.append(random.choice(self.candy_types))
        
        for candy in extra:
            inventory[candy] += 1
            
        cost_description = f"usando {self.candies_per_type_for_lollipop} de cada tipo"
        steps.append(f"Se hizo 1 chupetín ({cost_description}) y se recibieron {self.extra_candies_from_making} caramelos extra: {extra}")
        return 1
    
    def sell_lollipop(self, inventory, steps):
        """Vende un chupetín para obtener caramelos"""
        missing = self.get_missing_candies(inventory)
        chosen = []
        
        for candy, amount in missing.items():
            chosen.extend([candy] * min(amount, self.candies_from_selling - len(chosen)))
        
        while len(chosen) < self.candies_from_selling:
            chosen.append(random.choice(self.candy_types))
        
        for candy in chosen:
            inventory[candy] += 1
            
        steps.append(f"Se vendió 1 chupetín para recibir {self.candies_from_selling} caramelos: {chosen}")
    
    def get_missing_candies(self, inventory):
        """Calcula qué caramelos faltan para la próxima combinación"""
        missing = {}
        for candy_type in self.candy_types:
            missing[candy_type] = max(0, self.candies_per_type_for_lollipop - inventory[candy_type])
        return dict(sorted(missing.items(), key=lambda x: -x[1]))
    
    def simulate_game(self, max_iterations=1000, custom_distribution=None):
        """Simula un juego completo con distribución opcional personalizada"""
        steps = []
        
        # Reparto inicial - usar custom_distribution si se proporciona
        if custom_distribution:
            people_candies = custom_distribution
        else:
            people_candies = [random.choices(self.candy_types, k=self.candies_per_person) 
                              for _ in range(self.num_people)]
        
        all_candies = [c for pair in people_candies for c in pair]
        inventory = Counter(all_candies)
        
        lollipops = 0
        exchanges = 0
        iterations = 0
        
        steps.append("▶ Configuración del juego:")
        steps.append(f"Tipos de caramelos: {self.candy_types}")
        steps.append(f"Personas: {self.num_people}")
        steps.append(f"Caramelos por persona: {self.candies_per_person}")
        steps.append(f"Caramelos necesarios por chupetín: {self.candies_per_lollipop} ({self.candies_per_type_for_lollipop} de cada tipo)")
        steps.append(f"Caramelos obtenidos al vender: {self.candies_from_selling}")
        steps.append(f"Caramelos extra al hacer: {self.extra_candies_from_making}")
        steps.append("")
        
        steps.append("▶ Reparto inicial:")
        for i, candies in enumerate(people_candies, 1):
            steps.append(f"Persona {i}: {candies}")
        steps.append(f"Inventario inicial: {dict(inventory)}")
        steps.append("")
        
        # Hacer chupetines mientras se pueda
        while self.can_make_lollipop(inventory):
            lollipops += self.make_lollipop(inventory, steps)
        
        # Si no alcanza, vender chupetines y continuar
        while lollipops < self.num_people and iterations < max_iterations:
            if lollipops == 0:
                steps.append("⛔ No se pueden hacer más chupetines ni vender.")
                break
                
            self.sell_lollipop(inventory, steps)
            lollipops -= 1
            exchanges += 1
            
            while self.can_make_lollipop(inventory):
                lollipops += self.make_lollipop(inventory, steps)
            
            iterations += 1
        
        # Resultados
        success = lollipops >= self.num_people
        
        return {
            'steps': steps,
            'lollipops': lollipops,
            'exchanges': exchanges,
            'success': success,
            'iterations': iterations,
            'final_inventory': dict(inventory),
            'people_candies': people_candies
        }

def create_manual_distribution_interface(candy_types, num_people, candies_per_person):
    """Crea la interfaz para reparto manual de caramelos"""
    st.subheader("🎯 Reparto Manual de Caramelos")
    st.info(f"Asigna {candies_per_person} caramelos a cada una de las {num_people} personas")
    
    # Inicializar distribución en session_state si no existe
    if 'manual_distribution' not in st.session_state:
        st.session_state.manual_distribution = [[] for _ in range(num_people)]
    
    # Verificar si el número de personas cambió
    if len(st.session_state.manual_distribution) != num_people:
        st.session_state.manual_distribution = [[] for _ in range(num_people)]
    
    # Botones para generar distribuciones automáticas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎲 Generar Aleatorio"):
            st.session_state.manual_distribution = [
                random.choices(candy_types, k=candies_per_person) 
                for _ in range(num_people)
            ]
    
    with col2:
        if st.button("⚖️ Distribución Balanceada"):
            # Intentar distribuir de manera más equitativa
            total_candies = num_people * candies_per_person
            candies_per_type = total_candies // len(candy_types)
            remaining = total_candies % len(candy_types)
            
            # Crear pool de caramelos balanceado
            candy_pool = []
            for i, candy_type in enumerate(candy_types):
                count = candies_per_type + (1 if i < remaining else 0)
                candy_pool.extend([candy_type] * count)
            
            random.shuffle(candy_pool)
            
            # Distribuir entre personas
            st.session_state.manual_distribution = []
            for i in range(num_people):
                start_idx = i * candies_per_person
                end_idx = start_idx + candies_per_person
                st.session_state.manual_distribution.append(candy_pool[start_idx:end_idx])
    
    with col3:
        if st.button("🧹 Limpiar Todo"):
            st.session_state.manual_distribution = [[] for _ in range(num_people)]
    
    # Interfaz para cada persona
    distribution_valid = True
    
    for i in range(num_people):
        st.write(f"**Persona {i+1}:**")
        
        # Crear columnas para los selectores
        cols = st.columns(candies_per_person + 1)
        
        # Asegurar que la lista tenga el tamaño correcto
        while len(st.session_state.manual_distribution[i]) < candies_per_person:
            st.session_state.manual_distribution[i].append(candy_types[0])
        
        # Selectores para cada caramelo
        for j in range(candies_per_person):
            with cols[j]:
                current_value = st.session_state.manual_distribution[i][j]
                new_value = st.selectbox(
                    f"Caramelo {j+1}",
                    candy_types,
                    index=candy_types.index(current_value) if current_value in candy_types else 0,
                    key=f"person_{i}_candy_{j}"
                )
                st.session_state.manual_distribution[i][j] = new_value
        
        # Mostrar resumen de la persona
        with cols[-1]:
            person_counter = Counter(st.session_state.manual_distribution[i])
            st.write("**Resumen:**")
            for candy_type in candy_types:
                count = person_counter[candy_type]
                if count > 0:
                    st.write(f"{candy_type}: {count}")
    
    # Mostrar resumen total
    st.subheader("📊 Resumen Total")
    
    all_candies = [candy for person in st.session_state.manual_distribution for candy in person]
    total_counter = Counter(all_candies)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribución Total:**")
        for candy_type in candy_types:
            count = total_counter[candy_type]
            st.write(f"🍬 {candy_type.title()}: {count}")
    
    with col2:
        # Verificar si la distribución es válida
        total_candies = sum(len(person) for person in st.session_state.manual_distribution)
        expected_total = num_people * candies_per_person
        
        if total_candies == expected_total:
            st.success(f"✅ Distribución válida: {total_candies} caramelos")
        else:
            st.error(f"❌ Distribución inválida: {total_candies}/{expected_total} caramelos")
            distribution_valid = False
        
        # Mostrar balance
        if len(candy_types) > 1:
            min_count = min(total_counter[ct] for ct in candy_types)
            max_count = max(total_counter[ct] for ct in candy_types)
            balance = max_count - min_count
            
            if balance <= 1:
                st.info(f"⚖️ Muy balanceado (diff: {balance})")
            elif balance <= 3:
                st.warning(f"⚖️ Moderadamente balanceado (diff: {balance})")
            else:
                st.error(f"⚖️ Desbalanceado (diff: {balance})")
    
    return st.session_state.manual_distribution if distribution_valid else None

def optimize_with_optuna(base_config, n_trials=50, progress_bar=None):
    """Optimiza los parámetros del juego usando Optuna"""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está instalado. Instala con: pip install optuna")
    
    def objective(trial):
        # Parámetros a optimizar
        candies_per_lollipop = trial.suggest_int('candies_per_lollipop', 3, 15)
        candies_from_selling = trial.suggest_int('candies_from_selling', 4, 20)
        extra_candies_from_making = trial.suggest_int('extra_candies_from_making', 1, 8)
        
        # Crear modelo con parámetros optimizados
        model = CandyGameModel(
            candy_types=base_config['candy_types'],
            num_people=base_config['num_people'],
            candies_per_person=base_config['candies_per_person'],
            candies_per_lollipop=candies_per_lollipop,
            candies_from_selling=candies_from_selling,
            extra_candies_from_making=extra_candies_from_making
        )
        
        # Simular múltiples juegos para obtener estadísticas robustas
        successes = 0
        total_exchanges = 0
        
        for _ in range(20):  # 20 simulaciones por trial
            result = model.simulate_game()
            if result['success']:
                successes += 1
            total_exchanges += result['exchanges']
        
        success_rate = successes / 20
        avg_exchanges = total_exchanges / 20
        
        # Función objetivo: maximizar tasa de éxito, minimizar intercambios
        return success_rate - (avg_exchanges * 0.01)  # Penalizar muchos intercambios
    
    study = optuna.create_study(direction='maximize')
    
    if progress_bar:
        for i in range(n_trials):
            study.optimize(objective, n_trials=1)
            progress_bar.progress((i + 1) / n_trials)
    else:
        study.optimize(objective, n_trials=n_trials)
    
    return study

def create_visualizations(results_data):
    """Crea visualizaciones con Plotly"""
    
    # Gráfico de distribución de intercambios
    exchange_counts = [r['exchanges'] for r in results_data]
    exchange_dist = Counter(exchange_counts)
    
    fig_exchanges = px.bar(
        x=list(exchange_dist.keys()),
        y=list(exchange_dist.values()),
        title="Distribución de Intercambios",
        labels={'x': 'Número de Intercambios', 'y': 'Frecuencia'},
        color=list(exchange_dist.values()),
        color_continuous_scale='Viridis'
    )
    
    # Gráfico de tasa de éxito
    successes = sum(1 for r in results_data if r['success'])
    success_rate = (successes / len(results_data)) * 100
    
    fig_success = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = success_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tasa de Éxito (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    return fig_exchanges, fig_success

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Juego de Caramelos 🍬",
        page_icon="🍭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("🍬 Juego de Caramelos - Simulador 🍭")
    st.markdown("### Modelado flexible y reparto manual")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración del Juego")
    
    # Parámetros del juego (ahora con inputs numéricos)
    num_people = st.sidebar.slider("Número de personas", 5, 50, 10)
    candies_per_person = st.sidebar.slider("Caramelos por persona", 1, 10, 2)
    
    # Variables clave que ahora puedes modificar con st.number_input
    candies_per_lollipop = st.sidebar.number_input(
        "Caramelos necesarios por chupetín", 
        min_value=3, max_value=20, value=6, step=1
    )
    candies_from_selling = st.sidebar.number_input(
        "Caramelos obtenidos al vender un chupetín", 
        min_value=3, max_value=25, value=6, step=1
    )
    extra_candies_from_making = st.sidebar.number_input(
        "Caramelos extra al hacer un chupetín", 
        min_value=1, max_value=10, value=2, step=1
    )
    
    # Tipos de caramelos
    st.sidebar.subheader("Tipos de Caramelos")
    candy_types = ['limon', 'huevo', 'pera']  # Podríamos hacer esto configurable también
    st.sidebar.write(f"Tipos: {', '.join(candy_types)}")
    
    # Crear modelo
    model = CandyGameModel(
        candy_types=candy_types,
        num_people=num_people,
        candies_per_person=candies_per_person,
        candies_per_lollipop=candies_per_lollipop,
        candies_from_selling=candies_from_selling,
        extra_candies_from_making=extra_candies_from_making
    )
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🎮 Simulación Individual", "🎯 Reparto Manual", "📊 Análisis Múltiple"])
    
    with tab1:
        st.header("Simulación Individual")
        
        if st.button("🎲 Simular Juego", type="primary"):
            with st.spinner("Simulando..."):
                result = model.simulate_game()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Log de Simulación")
                log_text = "\n".join(result['steps'])
                st.text_area("Pasos de la simulación:", log_text, height=400)
            
            with col2:
                st.subheader("📊 Resultados")
                
                # Métricas principales
                st.metric("🍭 Chupetines totales", result['lollipops'])
                st.metric("🔁 Intercambios realizados", result['exchanges'])
                
                # Estado final
                if result['success']:
                    st.success("✅ ¡Objetivo logrado!")
                else:
                    st.error("❌ No se logró el objetivo")
                
                # Inventario final
                st.subheader("📦 Inventario Final")
                for candy, count in result['final_inventory'].items():
                    st.write(f"**{candy.title()}:** {count}")
                
                # Reparto inicial
                st.subheader("👥 Reparto Inicial")
                people_df = pd.DataFrame([
                    {"Persona": i+1, "Caramelos": ", ".join(candies)}
                    for i, candies in enumerate(result['people_candies'])
                ])
                st.dataframe(people_df, use_container_width=True)
    
    with tab2:
        st.header("🎯 Reparto Manual de Caramelos")
        
        # Crear interfaz de reparto manual
        manual_distribution = create_manual_distribution_interface(
            candy_types, num_people, candies_per_person
        )
        
        if manual_distribution:
            st.subheader("🎮 Simular con Reparto Manual")
            
            if st.button("🚀 Simular con Distribución Manual", type="primary"):
                with st.spinner("Simulando con distribución manual..."):
                    result = model.simulate_game(custom_distribution=manual_distribution)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📝 Log de Simulación")
                    log_text = "\n".join(result['steps'])
                    st.text_area("Pasos de la simulación:", log_text, height=400, key="manual_log")
                
                with col2:
                    st.subheader("📊 Resultados")
                    
                    # Métricas principales
                    st.metric("🍭 Chupetines totales", result['lollipops'])
                    st.metric("🔁 Intercambios realizados", result['exchanges'])
                    
                    # Estado final
                    if result['success']:
                        st.success("✅ ¡Objetivo logrado!")
                    else:
                        st.error("❌ No se logró el objetivo")
                    
                    # Inventario final
                    st.subheader("📦 Inventario Final")
                    for candy, count in result['final_inventory'].items():
                        st.write(f"**{candy.title()}:** {count}")
                    
                    # Análisis de la distribución manual
                    st.subheader("📊 Análisis de Distribución")
                    all_candies = [c for person in manual_distribution for c in person]
                    counter = Counter(all_candies)
                    
                    for candy_type in candy_types:
                        count = counter[candy_type]
                        percentage = (count / len(all_candies)) * 100
                        st.write(f"**{candy_type.title()}:** {count} ({percentage:.1f}%)")
    
    with tab3:
        st.header("Análisis de Múltiples Simulaciones")
        
        num_simulations = st.slider("Número de simulaciones", 10, 1000, 100)
        
        if st.button("🔄 Ejecutar Simulaciones Múltiples", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i in range(num_simulations):
                progress_bar.progress((i + 1) / num_simulations)
                status_text.text(f"Simulación {i + 1}/{num_simulations}")
                
                result = model.simulate_game()
                results.append(result)
            
            status_text.text("✅ Simulaciones completadas!")
            
            # Calcular estadísticas
            successes = sum(1 for r in results if r['success'])
            success_rate = (successes / num_simulations) * 100
            exchanges = [r['exchanges'] for r in results]
            avg_exchanges = sum(exchanges) / num_simulations
            
            # Mostrar métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Tasa de Éxito", f"{success_rate:.1f}%")
            
            with col2:
                st.metric("🔁 Promedio Intercambios", f"{avg_exchanges:.2f}")
            
            with col3:
                st.metric("📉 Min Intercambios", min(exchanges))
            
            with col4:
                st.metric("📈 Max Intercambios", max(exchanges))
            
            # Crear y mostrar visualizaciones
            fig_exchanges, fig_success = create_visualizations(results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_exchanges, use_container_width=True)
            with col2:
                st.plotly_chart(fig_success, use_container_width=True)
            
            # Tabla de distribución
            st.subheader("📊 Distribución Detallada")
            exchange_dist = Counter(exchanges)
            dist_df = pd.DataFrame([
                {
                    "Intercambios": exchanges,
                    "Frecuencia": count,
                    "Porcentaje": f"{(count/num_simulations)*100:.1f}%"
                }
                for exchanges, count in sorted(exchange_dist.items())
            ])
            st.dataframe(dist_df, use_container_width=True)

if __name__ == "__main__":
    main()