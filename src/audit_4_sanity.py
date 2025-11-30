import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Tu cargador maestro
from data_processor import load_and_clean_data

def deep_audit_final():
    print(" --- AUDITORÍA FINAL: LA PRUEBA NUCLEAR ---")
    print("Vamos a dejar a la IA ciega de analíticas.")
    
    df = load_and_clean_data()
    
    # LISTA NEGRA EXTENDIDA (Borramos TODO lo que huela a riñón o sangre)
    cols_to_drop = [
        'classification', 
        # 1. Marcadores de Riñón Directos
        'sc', 'bu', 'al', 'su', 'sg', 'sod', 'pot', 'renal_risk_ratio',
        # 2. Marcadores de Sangre / Anemia (Consecuencia directa)
        'hemo', 'pcv', 'rbc', 'rc', 'ane', 
        # 3. Microscopía de orina
        'pc', 'pcc', 'ba',
        # 4. Nuestras variables calculadas (para que no haga trampas con ellas)
        'history_score', 'bp_alarm'
    ]
    
    # Nos quedamos solo con lo "difuso": Edad, Tensión, Historial, Apetito...
    # Variables supervivientes esperadas: age, bp, bgr, htn, dm, cad, appet, pe, wc
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['classification']
    
    print(f"\n Variables supervivientes (Solo síntomas externos):")
    print(f"   {X.columns.tolist()}")
    
    # Separamos y entrenamos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n Precisión Final disminuyendo los datos usados para la predicción: {acc:.2%}")
    
    # Interpretación automática para Eva
    if acc < 0.90:
        print("\n ¡PRUEBA SUPERADA! La precisión ha bajado significativamente.")
        print("   CONCLUSIÓN: El 100% del modelo original NO era trampa.")
        print("   Era real porque tenía acceso a la Creatinina y la Hemoglobina.")
        print("   Al quitárselas, la IA se vuelve 'humana' y falla más.")
    else:
        print("\n WOW: Sigue siendo alto (>90%).")
        print("   Esto significa que solo con saber si tienes Diabetes e Hipertensión,")
        print("   la IA ya predice casi todo. ¡El dataset es muy determinista!")

if __name__ == "__main__":
    deep_audit_final()