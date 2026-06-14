# Renal Care AI — Detección Temprana de Enfermedad Renal Crónica (ERC)

Hackathon Boehringer Ingelheim 2025 — Equipo 5 "Debug Queens"

---

## Introducción

Con este proyecto buscamos desarrollar una herramienta clínica inteligente capaz de detectar de manera precoz la enfermedad renal crónica.
El objetivo no es solo diagnosticar el estado actual del paciente, sino hacer *Risk Scoring*: identificar perfiles aparentemente estables que tienen alta probabilidad de desarrollar ERC si no se interviene a tiempo.

---

## Descripción del proyecto

El sistema combina dos partes principales:

- **Un modelo de inteligencia artificial**: entrenado con Gradient Boosting (XGBoost) para clasificar el riesgo renal (bajo, moderado o alto) a partir de datos clínicos.
- **Una aplicación en Streamlit**: interfaz para que el profesional sanitario pueda ver el riesgo de cada paciente de manera visual, con la lógica del semáforo y los factores determinantes de cada predicción.

### ¿Qué hace?

- **Carga de datos**: el médico sube un CSV de pacientes (en producción se conectaría al EHR del hospital).
- **Selección de paciente**: se elige un paciente por su CIP/ID.
- **Diagnóstico con lógica de semáforo**: clasifica el riesgo en **Bajo / Medio / Alto** con la acción clínica recomendada.
- **Explicabilidad local**: muestra los *Factores Determinantes* que han influido en la predicción de ese paciente concreto (multipatología, ratio renal, anemia, edema, hipertensión descontrolada...).
- **Plan de seguimiento**: simula la evolución proyectada de la creatinina para ilustrar la gestión longitudinal.
- **Modo Auditoría** (página oculta): activa el dashboard de validación técnica con Curva ROC, Matriz de Confusión, Curva de Aprendizaje y Permutation Importance.

---

## Estructura del repositorio

```
Equipo-equipo-debug-queens-5/
│
├── memoria_proyecto/
│   └── Memoria_Proyecto.PDF
│
├── data/
│   ├── clean_data_.csv
│   ├── data.csv
│   └── reglas_descubiertas_ia.txt
│
├── demo/
│   ├── PAC1.png
│   ├── ...
│   └── PAC2.png
│
├── plots/
│   ├── 1_balance_clases.png
│   ├── ...
│   ├── 8_learning_curve.png
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── stress_audit2
│   └── Result_training1.png
│
├── model_xgb.pkl         
│
├── src/
│   │   # El núcleo 
│   ├── app.py                     # La web con Streamlit
│   ├── data_processor.py          # Limpiar y procesar la base de datos
│   ├── data_validation_plots.py   # Genera los plots para entender el dataset de entrenamiento
│   ├── train.py                   # Genera el model_xgb.pkl
│   ├── utils_demo_data.py         # Genera casos más representativos
│   │
│   │   # Data Science and Validation
│   ├── analysis_eda_basic.py      # Gráficos de datos (barras, heatmap)
│   ├── audit_1_metrics.py         # Matriz de Confusión y Curva ROC
│   ├── audit_2_learning_curves.py # Detección de overfitting
│   ├── audit_3_robustness.py      # Importancia de variables (Permutation)
│   ├── audit_4_sanity_check.py    # Prueba "no invasiva" (sin biomarcadores clave)
│   ├── audit_5_rules.py           # Reglas de texto del árbol
│   └── audit_6_explainability.py  # SHAP y confianza del modelo
│
├── requirements.txt
├── LICENSE
└── README.md

```
---

## Plan técnico del proyecto

Se usa la base de datos **UCI CKD** con 400 pacientes y 25 variables, con una distribución de clases de aproximadamente **60% enfermos / 40% sanos** — un balance natural favorable para el entrenamiento.

### Variables controladas

| Variable | Descripción |
|----------|-------------|
| `history_score` | Score de comorbilidad = Hipertensión + Diabetes + Enf. Coronaria |
| `bp_alarm` | Bandera roja si la presión arterial > 140 mmHg **o** hay diagnóstico de hipertensión |
| `renal_risk_ratio` | Variable de interacción = Creatinina Sérica × (Albúmina + 1) |

### Modelo y métricas

- **Algoritmo**: XGBoost (Gradient Boosting sobre árboles de decisión).
- **Por qué XGBoost**: maneja valores nulos de forma nativa, captura no linealidades y umbrales clínicos, ofrece *Feature Importance* (caja blanca) y es eficiente con datasets pequeños.
- **Validación**: Stratified K-Fold (k=5), manteniendo la proporción de clases en cada partición.
- **Métrica prioritaria**: **Recall (sensibilidad)** — minimizar falsos negativos es la prioridad clínica. Se monitoriza el **F1-Score** para evitar exceso de falsas alarmas.

Las variables más decisivas según el análisis son `renal_risk_ratio`, `hemo` (hemoglobina) y `sg` (gravedad específica), coherentes con los biomarcadores clínicos clave.

---

## Cómo empezar

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/TU-ORG/Hackaton-Boehringer-Ingelheim-2025.git
   cd Hackaton-Boehringer-Ingelheim-2025
   ```

2. **Instala las dependencias** (Python 3.9+, recomendado en un entorno virtual):
   ```bash
   python -m venv venv
   source venv/bin/activate        # En Windows: venv\Scripts\activate

   pip install -r requirements.txt
   ```
   Si no tienes `requirements.txt`, instala manualmente:
   ```bash
   pip install streamlit streamlit-option-menu pandas numpy joblib xgboost scikit-learn matplotlib seaborn shap
   ```

3. **Entrena el modelo** (genera `model_xgb.pkl`):
   ```bash
   python src/train.py
   ```

4. **Lanza la aplicación web:**
   ```bash
   streamlit run src/app.py
   ```

5. **Genera datos de demo y gráficos de auditoría (opcional):**
   ```bash
   python src/utils_demo_data.py          # Casos representativos para la demo
   python src/analysis_eda_basic.py       # Gráficos EDA
   python src/audit_1_metrics.py          # ROC + Matriz de Confusión
   python src/audit_3_robustness.py       # Permutation Importance
   python src/audit_6_explainability.py   # SHAP
   ```

---

## Interfaz de usuario

Para ejecutar la interfaz, usa el comando:

```bash
streamlit run src/app.py
```

Se abrirá en `http://localhost:8501`. Ve a **Consulta Médica**, sube un CSV de pacientes y selecciona uno por su CIP/ID. Activa el toggle **"Modo Auditoría"** en la barra lateral para ver el dashboard de validación técnica.

---

## IMPORTANTE

Toda la memoria y explicación del proyecto en detalle, tanto el marco médico como el técnico, se encuentra en el PDF de la carpeta **`memoria_proyecto/`**.

---

## Aviso importante

Este proyecto es un **prototipo desarrollado en un hackathon** con fines demostrativos y educativos.

El rendimiento muy alto observado (cercano al 100% de acierto) corresponde a este dataset concreto, que es especialmente limpio y separable. En un entorno clínico real —con mayor variabilidad entre pacientes, registros incompletos y diferencias entre laboratorios— **es esperable un rendimiento notablemente inferior**.

---
## Autoras
Eva Matabosch --> Ingeniería Biomédica
Miriam Morales --> Inteligencia Artificial
