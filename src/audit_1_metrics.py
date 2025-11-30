import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os

# Tu cargador maestro
from data_processor import load_and_clean_data

def generate_validation_plots():
    print(" Generando Gráficos de Validación Avanzada (ROC y Matriz de Confusión)...")

    # 1. Cargar datos
    df = load_and_clean_data()
    X = df.drop(columns=['classification'])
    y = df['classification']
    
    # Directorio para guardar
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Configuración del modelo (La misma que en train.py)
    model_params = {
        'objective': 'binary:logistic',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'random_state': 42
    }

    # --- PREPARACIÓN: Acumular predicciones de 5 folds ---
    # Para hacer una matriz y ROC "honestas", sumamos lo que pasa en los 5 experimentos
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_real_total = []
    y_pred_total = []
    y_proba_total = []

    print(" Re-ejecutando validación para capturar datos gráficos...")
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(**model_params)
        clf.fit(X_train, y_train)
        
        # Guardamos predicciones
        y_real_total.extend(y_val)
        y_pred_total.extend(clf.predict(X_val))
        y_proba_total.extend(clf.predict_proba(X_val)[:, 1])

    # --- GRÁFICO 1: Matriz de Confusión Agregada ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_real_total, y_pred_total)
    
    # Usamos Seaborn para que quede más bonito que el default
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicho Sano', 'Predicho Enfermo'],
                yticklabels=['Real Sano', 'Real Enfermo'])
    plt.title('Matriz de Confusión Global (5-Fold CV)', fontsize=14)
    plt.ylabel('Verdad (Realidad)')
    plt.xlabel('Predicción del Modelo')
    
    path_cm = os.path.join(plots_dir, '4_matriz_confusion.png')
    plt.savefig(path_cm)
    print(f" Matriz de Confusión guardada en: {path_cm}")
    plt.close()

    # --- GRÁFICO 2: Curva ROC ---
    fpr, tpr, thresholds = roc_curve(y_real_total, y_proba_total)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Línea de azar
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad/Recall)')
    plt.title('Curva ROC - Capacidad de Diagnóstico', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    path_roc = os.path.join(plots_dir, '5_curva_roc.png')
    plt.savefig(path_roc)
    print(f" Curva ROC guardada en: {path_roc}")
    plt.close()

if __name__ == "__main__":
    generate_validation_plots()