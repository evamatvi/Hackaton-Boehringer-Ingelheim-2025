import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from data_processor import load_and_clean_data

def audit_model():
    print("AUDITORÍA DE OVERFITTING Y REGLAS")
    
    # 1. Cargar y separar
    df = load_and_clean_data()
    X = df.drop(columns=['classification'])
    y = df['classification']

    # Separación Train/Test (70% / 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 2. Entrenar modelo "transparente" (poca profundidad)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    # 3. Calcular Notas
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n Train Accuracy: {train_acc:.2%}")
    print(f" Test Accuracy:  {test_acc:.2%}")

    # --- GENERAR EVIDENCIA 1: GRÁFICO DE BARRAS (La prueba visual) ---
    print("\n Generando gráfico de comparación...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Entrenamiento (Memorización)', 'Test (Generalización)'], y=[train_acc, test_acc], palette=['#e74c3c', '#2ecc71'])
    plt.ylim(0, 1.1)
    plt.title('Prueba de Robustez: ¿Memoriza o Aprende?', fontsize=14)
    plt.ylabel('Precisión (Accuracy)')
    
    # Añadir el número encima de la barra
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold', fontsize=12)

    save_path_img = os.path.join(plots_dir, '6_auditoria_overfitting.png')
    plt.savefig(save_path_img)
    print(f" Gráfico guardado en: {save_path_img}")
    plt.close()

    # --- GENERAR EVIDENCIA 2: REGLAS MÉDICAS (Archivo de Texto) ---
    print("\n Guardando las reglas descubiertas en texto...")
    
    # Extraer las reglas del árbol
    reglas = model.get_booster().get_dump()[0]
    
    data_dir = os.path.join(current_dir, '..', 'data')
    save_path_txt = os.path.join(data_dir, 'reglas_descubiertas_ia.txt')
    
    with open(save_path_txt, "w") as f:
        f.write("AUDITORÍA DE LÓGICA INTERNA (XGBoost)\n")
        f.write("=======================================\n")
        f.write("Este archivo muestra cómo toma decisiones el modelo.\n")
        f.write("Formato: [Variable < Valor] yes=... no=...\n\n")
        f.write(reglas)
        f.write("\n\nINTERPRETACIÓN:\n")
        f.write("- Si ves 'renal_risk_ratio < 1.3', significa que esa es la regla de oro.\n")
        f.write("- Si ves 'hemo < 13', confirma la anemia como factor clave.\n")
        
    print(f" Informe de texto guardado en: {save_path_txt}")

if __name__ == "__main__":
    audit_model()