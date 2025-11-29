import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Importamos vuestro procesador de datos maestro
from data_processor import load_and_clean_data

def train_model():
    print("INICIANDO ENTRENAMIENTO DEL MODELO DE IA")

    # 1. Cargar datos limpios
    df = load_and_clean_data()
    if df is None:
        return

    # 2. Separar Features (X) y Target (y)
    target_col = 'classification'
    # Quitamos la columna objetivo para tener solo los datos de entrada
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n Datos listos: {X.shape[0]} pacientes con {X.shape[1]} variables.")

    # 3. Configuración del Algoritmo (XGBoost)
    # NOTA: Como la clase 1 (Enfermos) es mayoría (60%), no necesitamos 'scale_pos_weight' alto.
    # El modelo aprenderá naturalmente a detectarlos.
    
    model_params = {
        'objective': 'binary:logistic',
        'n_estimators': 150,      # Número de árboles (cerebros pensando a la vez)
        'learning_rate': 0.05,    # Velocidad (lento pero seguro)
        'max_depth': 4,           # Profundidad (evita memorizar)
        'eval_metric': 'logloss', # Métrica de error interna
        'use_label_encoder': False,
        'n_jobs': -1,             # Usar toda la potencia del PC
        'random_state': 42
    }

    # 4. Validación Cruzada Estratificada (Stratified K-Fold)
    # Esto es lo que prometimos en el punto 4.2 de la documentación
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    recalls = []
    f1_scores = []
    cm_total = np.zeros((2, 2)) # Para sumar las matrices de confusión
    
    print(f"\n Ejecutando Validación Cruzada ({k} iteraciones)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Dividir datos
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar modelo temporal
        clf = xgb.XGBClassifier(**model_params)
        clf.fit(X_train, y_train)
        
        # Predecir
        y_pred = clf.predict(X_val)
        
        # Métricas
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        recalls.append(rec)
        f1_scores.append(f1)
        
        # Acumular matriz de confusión
        cm_total += confusion_matrix(y_val, y_pred)
        
        print(f"   [Fold {fold+1}] Recall (Sensibilidad): {rec:.2%} | F1-Score: {f1:.2%}")

    # 5. Resultados Globales
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    
    print("\n RESULTADOS FINALES")
    print(f"   Recall Promedio: {avg_recall:.2%} (Objetivo: >90%)")
    print(f"   F1-Score Promedio: {avg_f1:.2%}")
    
    # 6. Entrenar el Modelo Definitivo (Con TODOS los datos)
    print("\n Entrenando y guardando el 'Cerebro' final...")
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X, y)
    
    # Guardar en archivo .pkl (Esto es lo que cargará la web/demo)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'model_xgb.pkl')
    joblib.dump(final_model, model_path)
    print(f" Modelo guardado en: {model_path}")

    # 7. Gráfico de Feature Importance (Para Eva)
    print(" Generando gráfico de Importancia de Variables...")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(final_model, max_num_features=15, height=0.5, title="Top Variables Clave (Según IA)", color='teal', importance_type='weight')
    plt.grid(False)
    plt.show() # Esto abrirá una ventana con el gráfico

if __name__ == "__main__":
    train_model()