[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/_Fj1ryTe)
# Hackathon Boehringer – Repositorio del Equipo

Este repositorio se ha generado a partir de la **plantilla oficial** de la hackathon.

## ⚖️ Licencia
Este proyecto se distribuye bajo **Apache License 2.0**. Las contribuciones incluyen una concesión de derechos de **patentes** asociadas al código aportado, según los términos de la licencia.
Consulta el archivo [`LICENSE`](LICENSE) para más detalle.

## 📂 Estructura del repositorio
```
Equipo-equipo-debug-queens-5/
│
├── memoria_proyecto//                  
│   ├── Memoria_Proyecto.PDF
├── data/                  
│   ├── clean_data_.csv
│   ├── data.csv
│   └── reglas_descubiertas_ia.txt
│
├── demo/                  
│   ├── PAC1.png
│   ├── ...
│   └── PAC2.png
├── plots/                 
│   ├── 1_balance_clases.png
│   ├── ...
│   ├── 8_learning_curve.png
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── stress_audit2
│   └── Result_training1.png
│
├── model_xgb.pkl          <-- El "CEREBRO"
│
├── src/                   
│   │
│   │   # --- EL NÚCLEO (Core - La App Funcional) ---
│   ├── app.py (La web con Streamlit)
│   ├── data_processor.py  (Limpiar y procesar base de datos)
│   ├── data_validation_plots.py (General los plots para entender dataset de entrenamiento) 
│   ├── train.py  (Genera el modelo_xgb.pkl)
│   ├── utils_demo_data.py  (Genera casos mas representativos)
│   │
│   │   # --- LOS AUDITORES (Data Science & Validation) ---
│   ├── analysis_eda_basic.py      <-- (Era visualization.py) Gráficos de Datos (Barras, Heatmap)
│   ├── audit_1_metrics.py         <-- (Era validation_plots.py) Matriz Confusión y ROC
│   ├── audit_2_learning_curves.py <-- (Era plot_learning_curve.py) Detección de Overfitting
│   ├── audit_3_robustness.py      <-- (Era stress_test.py) Importancia de Variables (Cajitas)
│   ├── audit_4_sanity_check.py    <-- (Era deep_audit_final.py) Prueba "No Invasiva"
│   ├── audit_5_rules.py           <-- (Era audit_overfitting.py) Reglas de texto del árbol
│   └── audit_6_explainability.py  <-- (Era model_analysis.py) SHAP y Confianza
│
└── README.md

```
## Introducción

Con este proyecto buscamos desarrollar una herramienta clínica inteligente capaz de detectar de manera precoz la enfermedad renal crónica.

## Descripción del proyecto

El sistema combina dos partes principales:
Un modelo de inteligencia artificial: Entrenado con Gradient Boosting para clasificar el riesgo renal (bajo, moderado o alto) a partir de datos clínicos.
Una aplicación en Streamlit, que sirve como interfaz para que el profesional sanitario pueda ver el riesgo de cada paciente de manera visual.

## Plan técnico del proyecto
Se usa la base de datos UCI CKD con 400 pacientes y 25 variables con una distribución de clases de 60% enfermos y 40% sanos.

## Interfaz usuario
Para ejecutar la interfaz del usuario, se debe usar el comando streamlit run src/app.py

# IMPORTANTE: 
Toda la memoria y explicación del proyecto en detalle, tanto el marco médico como técnico, se encuentra en el PDF de la carpeta memoria_proyecto/

## 🚀 Cómo empezar
1. Clona el repositorio:
   ```bash
   git clone https://github.com/TU-ORG/equipo-nombre.git
   ```
2. Instala dependencias (elige tu stack):
   - **Python**:
     ```bash
     pip install -r requirements.txt  
     pip install pytest
     ```
   - **Node**:
     ```bash
     npm ci
     ```
3. Ejecuta tests:
   - **Python**: `pytest -q`
   - **Node**: `npm test`

## 🧪 Criterios de evaluación (orientativo)
- Claridad del problema y solución
- Calidad técnica (código, tests, CI)
- Demo funcional / UX
- Impacto y viabilidad
- Presentación / pitch

## 🔐 Seguridad y datos
- No subas secretos (tokens, claves) ni datos sensibles.
- Usa variables de entorno y `.env` (ignorado por Git).

## 📦 Entrega final
- Crear un **Release** con etiqueta `v1.0-hackathon` con README actualizado, instrucciones y demo.
- Powerpoint proyecto final
- Mockup si aplica

## 👥 Créditos
Incluye autores del equipo y mentores si aplica.
