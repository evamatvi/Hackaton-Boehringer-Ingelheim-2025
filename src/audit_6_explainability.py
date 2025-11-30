import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Importamos el cargador de datos
from data_processor import load_and_clean_data

def analyze_model():
    print("INICIANDO AUDITORÍA DEL MODELO")
    
    # 1. Cargar el modelo entrenado
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'model_xgb.pkl')
    
    try:
        model = joblib.load(model_path)
        print("Modelo cargado correctamente.")
    except FileNotFoundError:
        print("Error: No encuentro el modelo. Ejecuta train.py primero.")
        return

    # 2. Cargar datos para probar
    df = load_and_clean_data()
    X = df.drop(columns=['classification'])
    y = df['classification']

    # 3. ANÁLISIS DE PROBABILIDADES (La prueba del algodón)
    # Obtenemos la probabilidad (0.0 a 1.0) en lugar de la clase (0 o 1)
    probs = model.predict_proba(X)[:, 1]
    
    # Creamos un DataFrame para ver los casos límite
    analysis_df = X.copy()
    analysis_df['Realidad'] = y
    analysis_df['Probabilidad_IA'] = probs
    
    print("\n Distribución de Confianza")
    # Si la IA es buena, las probabilidades deberían estar cerca de 0 o de 1.
    # Si duda mucho, estarán por el 0.4 - 0.6
    plt.figure(figsize=(10, 6))
    sns.histplot(probs, bins=20, kde=True, color='purple')
    plt.title('¿Cómo de segura está la IA? (Histograma de Probabilidades)')
    plt.xlabel('Probabilidad de Riesgo (0=Seguro Sano, 1=Seguro Enfermo)')
    plt.ylabel('Cantidad de Pacientes')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.show()

    # 4. EXPLICABILIDAD AVANZADA (SHAP)
    print("\n Generando Gráficos SHAP (Esto puede tardar un poco)")
    
    # Creamos el explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Gráfico Resumen (Beeswarm) - EL MEJOR GRÁFICO PARA MÉDICOS
    # Muestra: Variables importantes + Si valores altos/bajos suben/bajan el riesgo
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Impacto Clínico de Variables (SHAP)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 5. SIMULACIÓN DE OUTPUT PARA LA WEB
    print("\n Simulando Output para la Demo Web")
    # Cogemos al paciente número 0 como ejemplo
    paciente_ejemplo = X.iloc[[0]] 
    riesgo = model.predict_proba(paciente_ejemplo)[0, 1]
    prediccion = int(riesgo > 0.5)
    
    output_json = {
        "prediccion_clase": prediccion, # 0 o 1
        "probabilidad_riesgo": round(riesgo * 100, 2), # 99.5%
        "nivel_alerta": "CRÍTICO" if riesgo > 0.8 else ("ALTO" if riesgo > 0.5 else "BAJO"),
        "explicacion": "El paciente supera el umbral del 50% de probabilidad."
    }
    
    print(f"Ejemplo de respuesta para el Front-end:\n{output_json}")

if __name__ == "__main__":
    analyze_model()