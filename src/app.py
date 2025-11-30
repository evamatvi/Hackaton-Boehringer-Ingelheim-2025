import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_option_menu import option_menu

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Boehringer - Detección ERC",
    page_icon="🏥",
    layout="wide"
)

# --- 1. CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'model_xgb.pkl')
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Error: No se encuentra 'model_xgb.pkl'.")
        return None

model = load_model()

# --- 2. FUNCIÓN DE LIMPIEZA EN VIVO ---
def preprocess_uploaded_file(uploaded_file):
    try:
        # INTENTO 1: Leer normal
        df = pd.read_csv(uploaded_file)
        
        # INTENTO 2: Si detectamos formato Excel (punto y coma)
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')

        # 1. Limpieza básica
        df.replace('?', np.nan, inplace=True)
        cols_texto = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        for col in cols_texto:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
                df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan})

        # 2. Convertir a números
        cols_num = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Mapeo a Binario (0/1)
        mapa = {'yes': 1, 'no': 0, 'present': 1, 'notpresent': 0, 'abnormal': 1, 'normal': 0, 'poor': 1, 'good': 0}
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'id': 
                df[col] = df[col].map(mapa).fillna(df[col])

        # 4. FEATURE ENGINEERING
        if {'htn', 'dm', 'cad'}.issubset(df.columns):
            df['history_score'] = df['htn'].fillna(0) + df['dm'].fillna(0) + df['cad'].fillna(0)
        else:
            df['history_score'] = 0

        if {'bp', 'htn'}.issubset(df.columns):
            df['bp_alarm'] = ((df['bp'] > 140) | (df['htn'] == 1)).astype(int)
        else:
            df['bp_alarm'] = 0

        if {'sc', 'al'}.issubset(df.columns):
            df['renal_risk_ratio'] = df['sc'] * (df['al'].fillna(0) + 1)
        else:
            df['renal_risk_ratio'] = 0
        
        return df

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        return None

# --- 3. INTERFAZ DE USUARIO (Sidebar) ---

patient_df = None
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("Renal Care AI")
    st.caption("Hackathon Boehringer 2025")
    st.markdown("---")

    # MODO DESARROLLADOR
    dev_mode = st.toggle("Modo Auditoría")
    
    # Menú dinámico
    opciones = ["Inicio", "Consulta Médica"]
    iconos = ["house", "activity"]
    
    if dev_mode:
        opciones.append("Validación Técnica")
        iconos.append("graph-up-arrow")
    
    opciones.append("Equipo")
    iconos.append("people")

    selected = option_menu(
        menu_title=None,
        options=opciones,
        icons=iconos,
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#6a0dad"}}
    )
    
    # Uploader solo si estamos en Consulta
    if selected == "Consulta Médica":
        st.markdown("---")
        st.write("**Historia Clínica Digital**")
        uploaded_file = st.file_uploader("Cargar dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            with st.spinner('Procesando datos...'):
                patient_df = preprocess_uploaded_file(uploaded_file)
            st.success("Pacientes cargados")

# --- 4. PANEL PRINCIPAL ---
if selected == "Inicio":
    st.title("Bienvenido a Renal Care AI")
    st.markdown("### Herramienta de cribado inteligente para Atención Primaria.")
    st.markdown("### ¿Cómo funciona?")
    st.markdown("""
    1. **Sube el Dataset:** El sistema procesará automáticamente los datos brutos.
    2. **Busca al Paciente:** Introduce el CIP o ID.
    3. **Obtén el Diagnóstico:** La IA analizará biomarcadores y antecedentes en tiempo real.
    """)
    st.info("Ve a la pestaña **Consulta Médica** para comenzar.")

# --- PÁGINA 2: CONSULTA MÉDICA ---
if selected == "Consulta Médica":
    st.title("Consulta de Atención Primaria")
    
    if model is None:
        st.error("ERROR: No se encuentra el modelo IA. Ejecuta train.py primero.")
        st.stop()

    if patient_df is None:
        st.info("Por favor, carga el archivo de pacientes en el menú lateral.")
        st.stop()

if patient_df is not None:
    # Selector de Paciente
    st.markdown("### Buscador de Pacientes")
    
    if 'id' in patient_df.columns:
        patient_ids = patient_df['id'].tolist()
        # Buscamos columnas clave para mostrar en el selectbox si existen
        # (Opcional: podrías personalizar esto más)
    else:
        patient_ids = patient_df.index.tolist()
        
    selected_id = st.selectbox("Seleccionar CIP / ID Paciente:", patient_ids)
    
    # Extraer datos del paciente seleccionado
    if 'id' in patient_df.columns:
        p_data = patient_df[patient_df['id'] == selected_id].iloc[0]
    else:
        p_data = patient_df.iloc[selected_id]

    st.markdown("---")
    
    # --- MOSTRAR DATOS DEL PACIENTE (Ficha Clínica) ---
    st.subheader(f"Datos del paciente: {selected_id}")

    tab_clinica, tab_lab = st.tabs(["Historia", "Analítica"])
    
    with tab_clinica:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Edad", f"{int(p_data.get('age', 0))} años")
        col_b.metric("Presión Arterial", f"{p_data.get('bp', '?')} mmHg")
        
        st.markdown("---")
        htn_txt = "SÍ" if p_data.get('htn', 0) == 1 else "No"
        dm_txt = "SÍ" if p_data.get('dm', 0) == 1 else "No"
        pe_txt = "SÍ" if p_data.get('pe', 0) == 1 else "No"
        appet_txt = "Pobre" if p_data.get('appet', 0) == 1 else "Bueno"
        
        col_c.metric("Hipertensión / Diabetes", f"{htn_txt} / {dm_txt}")
        col_d.metric("Edema / Apetito", f"{pe_txt} / {appet_txt}")
        
    with tab_lab:
        col_lab1, col_lab2, col_lab3 = st.columns(3)
        col_lab1.metric("Creatinina (sc)", f"{p_data.get('sc', '?')} mg/dL")
        col_lab1.metric("Hemoglobina (hemo)", f"{p_data.get('hemo', '?')} g/dL")
        
        # Albúmina interpretada
        alb_val = p_data.get('al', 0)
        alb_texto = "Positivo" if alb_val > 0 else "Negativo"
        col_lab2.metric("Albúmina Orina (al)", f"{alb_texto} ({int(alb_val)})")
        col_lab2.metric("Glucosa (bgr)", f"{p_data.get('bgr', '?')} mg/dL")
        col_lab3.metric("Densidad Orina (sg)", f"{p_data.get('sg', '?')}")


    # --- BLOQUE DE AUDITORÍA DE DATOS BRUTOS ---
    with st.expander("Ver Registro Completo"):
        
        # 1. Definir la lista de variables a EXCLUIR (Ingeniería + Objetivo)
        cols_a_excluir = [
            'history_score', 'bp_alarm', 'renal_risk_ratio', 
            'classification', 'id'
        ]
        
        # 2. Filtrar la data del paciente, quitando las columnas de la Lista
        # Usamos .drop en el índice de la serie p_data
        p_data_raw = p_data.drop(labels=[c for c in cols_a_excluir if c in p_data.index], errors='ignore')
        
        # 3. Mostrar el resultado filtrado (transpuesto para mejor lectura)
        df_display = p_data_raw.to_frame().T.fillna('N/A')
        st.dataframe(df_display.T, use_container_width=True)
    
    st.markdown("---")
    
    # --- PREDICCIÓN DE LA IA ---
    st.markdown("### Análisis de Riesgo (IA)")
    
    try:
        # Preparar datos para el modelo (ordenar columnas)
        cols_modelo = model.get_booster().feature_names
        input_data = pd.DataFrame([p_data])
        for c in cols_modelo:
            if c not in input_data.columns: input_data[c] = 0
        
        input_data = input_data[cols_modelo]
        
        # Predecir Probabilidad
        prob = model.predict_proba(input_data)[0][1]
        
        c_res1, c_res2 = st.columns([1, 2])
        riesgo_nivel = "Bajo"

        with c_res1:
            st.subheader("Nivel de Alerta")
            # --- LÓGICA DE TRIAJE Y ELIMINACIÓN DE % ---
            if prob < 0.40:
                st.success("🟢 BAJO RIESGO")
                st.caption(f"Probabilidad estimada: {prob:.1%}")
                riesgo_nivel = "Bajo"
            else:
                # Si IA ve riesgo (>40%), miramos la creatinina para decidir gravedad (Alto vs Medio)
                if p_data.get('sc', 0) <= 1.3:
                    st.warning("🟠 RIESGO MEDIO")
                    st.metric("Alerta:", "Estadio Temprano", delta_color="off")
                    riesgo_nivel = "Medio"
                else:
                    st.error("🔴 ALTO RIESGO")
                    st.metric("Alerta:", "Daño Crítico", delta_color="inverse")
                    riesgo_nivel = "Alto"

        with c_res2:
            st.info("**Análisis de Factores:**")
            reasons = []
            
            # --- FACTORES DETERMINANTES (Lenguaje Clínico) ---
            
            # 1. Ratio Renal (El factor más importante)
            if p_data.get('renal_risk_ratio', 0) > 1.3:
                reasons.append(f"- **DAÑO FILTRANTE SEVERO:** Creatinina y Albúmina combinadas superan el umbral de riesgo (Ratio: {p_data['renal_risk_ratio']:.2f}).")
            elif p_data.get('renal_risk_ratio', 0) > 0.8:
                reasons.append(f"- **Ratio Renal EN ALERTA:** Valores limítrofes, sugiriendo inicio de daño ({p_data['renal_risk_ratio']:.2f}).")
                
            # 2. Historial
            if p_data.get('history_score', 0) >= 2:
                reasons.append(f"- **MULTIPATOLOGÍA:** Paciente con {int(p_data['history_score'])} antecedentes (HTN y/o DM) que incrementan el riesgo renal.")
            
            # 3. Anemia
            if p_data.get('hemo', 15) < 12.5:
                reasons.append(f"- **ANEMIA DETECTADA:** Hemoglobina baja ({p_data['hemo']}). Posible signo de fallo renal subclínico.")
            
            # 4. Presión
            if p_data.get('bp_alarm', 0) == 1:
                reasons.append("- **HIPERTENSIÓN:** Presión arterial alta o historial descontrolado.")

            # Imprimir razones
            if not reasons and riesgo_nivel == "Bajo":
                st.write("✅ Perfil estable. No se detectan anomalías en los biomarcadores.")
            elif not reasons:
                st.write("- Patrón complejo detectado por el algoritmo.")
            else:
                for r in reasons: st.write(r)

        # RECOMENDACIONES FINALES
        st.markdown("---")
        if riesgo_nivel == "Alto":
            st.error("**🚑 Acción:** Derivación urgente a Nefrología. Daño renal visible.")
        elif riesgo_nivel == "Medio":
            st.warning("**⚠️ Acción:** Vigilancia activa (repetir analítica en 3 meses). Control estricto de TA/Glucosa.")
        else:
            st.success("**✅ Acción:** Control rutinario anual.")

    except Exception as e:
        st.error(f"Error en predicción: {e}")

    
# --- PÁGINA OCULTA: VALIDACIÓN TÉCNICA ---
if selected == "Validación Técnica":
    st.title("Auditoría Técnica del Modelo")
    st.markdown("### Evidencias de robustez y rendimiento (XGBoost)")
    
    # Definimos la ruta a las imágenes
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', 'plots')
    
    # Pestañas para organizar los gráficos
    tab1, tab2, tab3 = st.tabs(["Diagnóstico (ROC)", "Overfitting", "Variables Clave"])
    
    with tab1:
        st.subheader("Capacidad de Separación (Sano vs Enfermo)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(os.path.join(plots_dir, '5_curva_roc.png'), caption="Curva ROC (AUC = 1.0)", use_container_width=True)
        with col2:
            st.image(os.path.join(plots_dir, '4_matriz_confusion.png'), caption="Matriz de Confusión (Cero Falsos Negativos)", use_container_width=True)

    with tab2:
        st.subheader("Curvas de Aprendizaje")
        st.write("La convergencia de las líneas roja (train) y verde (test) demuestra que el modelo generaliza bien y no memoriza.")
        st.image(os.path.join(plots_dir, '7_learning_curve.png'), use_container_width=True)

    with tab3:
        st.subheader("¿Qué mira la IA? (Permutation Importance)")
        st.write("Ranking de variables que más afectan a la decisión si son eliminadas.")
        st.image(os.path.join(plots_dir, '8_permutation_importance.png'), use_container_width=True)

# --- PÁGINA 4: EQUIPO ---
if selected == "Equipo":
    st.title("Debug Queens")
    st.markdown("### Proyecto desarrollado para el Hackathon Boehringer Ingelheim 2025")
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Eva**\n\n*Ingeniería Biomédica*\n\nExpertise: Fisiología Renal, Datos Clínicos.")
    with col2:
        st.info("**Miriam**\n\n*Inteligencia Artificial*\n\nExpertise: Data cience, Machine Learning, XGBoost.")