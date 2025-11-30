import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import learning_curve
import os

# Tu cargador maestro
from data_processor import load_and_clean_data

def generate_learning_curve():
    print(" Generando Curva de Aprendizaje (Learning Curve)...")
    
    # 1. Cargar datos
    df = load_and_clean_data()
    X = df.drop(columns=['classification'])
    y = df['classification']
    
    # 2. Configurar el modelo (El mismo de siempre)
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric='logloss', 
        use_label_encoder=False
    )
    
    # 3. Calcular la curva
    # Empezamos en 0.3 (30%) para evitar el error de "falta una clase"
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.3, 1.0, 5), 
        scoring='accuracy'
    )
    
    # Medias y desviación estándar (para la sombra)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # 4. Pintar el gráfico
    plt.figure(figsize=(10, 6))
    plt.title("Curva de Aprendizaje: ¿Overfitting o Robustez?", fontsize=14)
    plt.xlabel("Número de Pacientes (Training Examples)")
    plt.ylabel("Precisión (Accuracy)")
    plt.ylim(0.8, 1.02) # Zoom en la parte alta
    plt.grid(True)
    
    # Sombras (Desviación estándar)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    # Líneas principales
    plt.plot(train_sizes, train_mean, 'o-', color="#e74c3c", label="Entrenamiento (Memorización)")
    plt.plot(train_sizes, test_mean, 'o-', color="#2ecc71", label="Validación (Generalización)")
    
    plt.legend(loc="best")
    
    # 5. Guardar
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    save_path = os.path.join(plots_dir, '7_learning_curve.png')
    plt.savefig(save_path)
    print(f" Gráfico guardado en: {save_path}")
    plt.close() # Cerrar para no gastar memoria

if __name__ == "__main__":
    generate_learning_curve()