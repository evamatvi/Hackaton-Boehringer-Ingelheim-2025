import pandas as pd
import numpy as np
import os

def create_demo_file():
    print("🏭 Generando archivo de prueba para la DEMO...")
    
    # Rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    original_path = os.path.join(data_dir, 'data.csv')
    output_path = os.path.join(data_dir, 'demo_pacientes.csv')
    
    # 1. INTENTAR USAR DATOS REALES
    if os.path.exists(original_path):
        print(" Encontrado data.csv original.")
        try:
            df = pd.read_csv(original_path)
            
            # Borrar la columna de respuesta (Si existe)
            if 'classification' in df.columns:
                df = df.drop(columns=['classification'])
                print(" Columna 'classification' eliminada.")
            
            # Guardar versión limpia (separada por comas estándar)
            df.to_csv(output_path, index=False)
            print(f" Archivo guardado en: {output_path}")
            return
        except Exception as e:
            print(f" Error leyendo original: {e}. Pasando a modo sintético.")

    # 2. GENERAR DATOS SINTÉTICOS (Si falla lo anterior)
    print(" Creando pacientes sintéticos desde cero...")
    
    # Creamos 5 pacientes variados
    data = {
        'id': [101, 102, 103, 104, 105],
        'age': [35, 65, 55, 70, 28],
        'bp':  [70, 90, 150, 100, 60],   # Presión
        'sg':  [1.025, 1.010, 1.015, 1.010, 1.025],
        'al':  [0, 2, 1, 4, 0],          # Albúmina (Clave)
        'su':  [0, 0, 1, 3, 0],
        'rbc': ['normal', 'normal', 'normal', 'abnormal', 'normal'],
        'pc':  ['normal', 'normal', 'normal', 'abnormal', 'normal'],
        'pcc': ['notpresent', 'notpresent', 'notpresent', 'present', 'notpresent'],
        'ba':  ['notpresent', 'notpresent', 'notpresent', 'present', 'notpresent'],
        'bgr': [90, 150, 220, 280, 85],  # Glucosa
        'bu':  [30, 80, 45, 120, 25],    # Urea
        'sc':  [0.9, 3.5, 1.3, 5.2, 0.7],# Creatinina (Clave)
        'sod': [140, 132, 138, 128, 145],
        'pot': [4.5, 5.2, 4.1, 6.0, 3.9],
        'hemo':[15.5, 9.0, 12.5, 8.5, 16.0], # Hemoglobina (Clave)
        'pcv': [45, 28, 38, 25, 48],
        'wc':  [8000, 11000, 9000, 15000, 7500],
        'rc':  [5.5, 3.0, 4.2, 2.8, 6.0],
        'htn': ['no', 'yes', 'yes', 'yes', 'no'], # Hipertensión
        'dm':  ['no', 'yes', 'yes', 'yes', 'no'], # Diabetes
        'cad': ['no', 'no', 'no', 'yes', 'no'],
        'appet':['good', 'poor', 'good', 'poor', 'good'],
        'pe':  ['no', 'yes', 'no', 'yes', 'no'],
        'ane': ['no', 'yes', 'no', 'yes', 'no']
    }
    
    df_synth = pd.DataFrame(data)
    df_synth.to_csv(output_path, index=False)
    
    print(" ¡Éxito! 5 Pacientes generados:")
    print("   - Paciente 101: Sano")
    print("   - Paciente 102: Enferma (Marta)")
    print("   - Paciente 103: Riesgo (Luis)")
    print(f" Archivo guardado en: {output_path}")

if __name__ == "__main__":
    create_demo_file()