import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.inspection import permutation_importance
import os

# Tu cargador de confianza
from data_processor import load_and_clean_data

def plot_learning_curve(estimator, X, y, title="Curvas de Aprendizaje"):
    """
    Genera el gráfico diagnóstico de Overfitting vs Generalización.
    CORREGIDO: Empieza con más datos para evitar errores de clases faltantes.
    """
    # Empezamos en 0.3 (30%) para asegurar que hay Sanos y Enfermos en el grupo
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.3, 1.0, 5), 
        scoring='accuracy'
    )
    
    # Calculamos medias
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Ploteamos
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plt.xlabel("Número de Pacientes usados")
    plt.ylabel("Precisión (Accuracy)")
    plt.ylim(0.8, 1.02)
    plt.grid(True)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#e74c3c", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#2ecc71", label="Validación")
    
    plt.legend(loc="best")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '..', 'plots', '7_learning_curve.png')
    plt.savefig(save_path)
    print(f" Curva de Aprendizaje guardada en: {save_path}")
    plt.close()

def stress_test():
    print(" --- PRUEBAS DE ESTRÉS (Versión Segura) ---")
    
    df = load_and_clean_data()
    X = df.drop(columns=['classification'])
    y = df['classification']
    
    # Modelo
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, 
        eval_metric='logloss', use_label_encoder=False
    )
    
    # --- PRUEBA 1: CURVAS DE APRENDIZAJE ---
    print("\n Generando Curvas de Aprendizaje...")
    try:
        plot_learning_curve(model, X, y)
    except Exception as e:
        print(f" Aviso: No se pudo generar la curva (Error: {e})")
    
    # --- PRUEBA 2: PERMUTATION IMPORTANCE ---
    print("\n Ejecutando Permutation Importance (Ranking Real)...")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    
    r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = r.importances_mean.argsort()[::-1]
    
    print("\n Top 5 Variables CRÍTICAS (Las que rompen el modelo si faltan):")
    for i in sorted_idx[:5]:
        print(f"   - {X.columns[i]}: {r.importances_mean[i]:.4f} de impacto.")
        
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.boxplot(r.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importance (Validación Real)")
    plt.tight_layout()
    
    save_path_perm = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', '8_permutation_importance.png')
    plt.savefig(save_path_perm)
    print(f" Gráfico de Permutación guardado en: {save_path_perm}")
    plt.show()

if __name__ == "__main__":
    stress_test()