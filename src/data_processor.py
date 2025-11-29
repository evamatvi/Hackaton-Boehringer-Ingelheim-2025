import pandas as pd
import numpy as np
import os

def load_and_clean_data():
    """
    Carga 'data.csv', limpia errores de formato y genera nuevas variables (Feature Engineering).
    """
    # --- 1. CARGA ROBUSTA ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'data.csv')
    
    print(f"[INFO] Buscando archivo en: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[ERROR] No se encuentra 'data.csv'.")
        return None

    # --- 2. LIMPIEZA DE SUCIEDAD ---
    # Reemplazar '?' por NaN real
    df.replace('?', np.nan, inplace=True)
    
    # Limpiar columnas de TEXTO (quitando tabuladores y espacios)
    # IMPORTANTE: Usamos un método que no rompe los NaNs
    cols_texto = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    for col in cols_texto:
        if col in df.columns:
            # Convertimos a string, limpiamos, y volvemos a poner NaN donde había 'nan' literal
            df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
            df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan})

    # --- 3. CONVERSIÓN A NÚMEROS ---
    # Corregido: Nombres exactos del dataset Kaggle (wc, rc)
    cols_numericas = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 4. MAPEO A BINARIO (0 y 1) ---
    if 'classification' in df.columns:
        # Mapeamos incluyendo posibles errores de tipeo que queden
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0}).fillna(0)

    mapa_binario = {
        'yes': 1, 'no': 0,
        'present': 1, 'notpresent': 0,
        'abnormal': 1, 'normal': 0,
        'poor': 1, 'good': 0
    }
    
    # Aplicar mapa a las columnas categóricas restantes
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(mapa_binario)

    # --- 5. FEATURE ENGINEERING (Tu toque maestro) ---
    print("[INFO] Generando variables de Historia Clínica...")
    
    # A. Risk Score (Historial de enfermedades previas)
    # Llenamos con 0 solo para sumar (asumiendo que nulo = no tiene la enfermedad registrada)
    if {'htn', 'dm', 'cad'}.issubset(df.columns):
        df['history_score'] = df['htn'].fillna(0) + df['dm'].fillna(0) + df['cad'].fillna(0)
    
    # B. Alerta de Presión Arterial (Hipertensión Descontrolada)
    # Si bp > 140 o tiene diagnóstico de hipertensión
    if 'bp' in df.columns and 'htn' in df.columns:
        df['bp_alarm'] = ((df['bp'] > 140) | (df['htn'] == 1)).astype(int)

    # C. Ratio de Riesgo Renal (Creatinina x Albúmina)
    # Un indicador avanzado de daño
    if 'sc' in df.columns and 'al' in df.columns:
        df['renal_risk_ratio'] = df['sc'] * (df['al'].fillna(0) + 1)

    # Eliminar ID si existe (no sirve para predecir)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    print("[INFO] Proceso finalizado.")
    print(f"       Dimensiones: {df.shape}")
    
    return df

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("Modo Prueba: Ejecutando y guardando CSV de control")
    df_limpio = load_and_clean_data()
    
    if df_limpio is not None:
        print("\n[CHECK FINAL] Tipos de datos:")
        print(df_limpio.info())
        
        # --- AQUÍ ESTÁ EL CAMBIO ---
        # Guardamos el archivo limpio para que Eva pueda abrirlo en Excel y verificarlo
        # Lo guardamos en la carpeta data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, '..', 'data', 'clean_data_debug.csv')
        
        df_limpio.to_csv(output_path, index=False)
        print(f"\n Archivo de control guardado en: {output_path}")
        print("(Puedes abrirlo en Excel para revisarlo manualmente)")