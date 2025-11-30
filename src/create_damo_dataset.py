import pandas as pd
import numpy as np
import os

def generate_new_patients():
    print(" Generando pacientes NUEVOS (Simulación de Sala de Espera)...")
    
    # Definimos 5 pacientes estratégicos para la Demo
    data = [
        # 1. EL SANO (Para probar el Verde)
        # Joven, deportista, todo perfecto.
        {
            'id': 'PAC-001 (Pedro - Sano)',
            'age': 35, 'bp': 70, 'sg': 1.025, 'al': 0, 'su': 0, 
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 90, 'bu': 30, 'sc': 0.8, 'sod': 140, 'pot': 4.5, 'hemo': 15.5, 
            'pcv': 45, 'wc': 7000, 'rc': 5.5,
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        },
        
        # 2. LA ENFERMA GRAVE (Para probar el Rojo)
        # Mayor, diabética, anémica, creatinina alta.
        {
            'id': 'PAC-002 (Marta - Grave)',
            'age': 68, 'bp': 160, 'sg': 1.010, 'al': 3, 'su': 2, 
            'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
            'bgr': 250, 'bu': 90, 'sc': 4.2, 'sod': 130, 'pot': 5.8, 'hemo': 9.2, 
            'pcv': 30, 'wc': 12000, 'rc': 3.1,
            'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
        },

        # 3. EL CASO "LUIS" (El Truco de la IA - Riesgo Oculto)
        # Creatinina casi normal (1.3), pero Diabético + Hipertenso + Albúmina leve.
        # Aquí es donde vuestro 'history_score' y 'renal_risk_ratio' brillarán.
        {
            'id': 'PAC-003 (Luis - Oculto)',
            'age': 55, 'bp': 145, 'sg': 1.015, 'al': 1, 'su': 0, 
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 140, 'bu': 45, 'sc': 1.3, 'sod': 138, 'pot': 4.1, 'hemo': 12.8, 
            'pcv': 39, 'wc': 9000, 'rc': 4.5,
            'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        },

        # 4. EL ABUELO ESTABLE (Falso positivo evitado)
        # Mayor (80 años), Hipertenso, pero riñón funcionando bien (Creatinina ok).
        # Debería salir Verde o Riesgo muy bajo.
        {
            'id': 'PAC-004 (Abuelo - Estable)',
            'age': 80, 'bp': 130, 'sg': 1.020, 'al': 0, 'su': 0, 
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 100, 'bu': 50, 'sc': 1.1, 'sod': 135, 'pot': 4.8, 'hemo': 13.5, 
            'pcv': 41, 'wc': 6000, 'rc': 4.2,
            'htn': 'yes', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        },
        
        # 5. LA JOVEN CON NEFRITIS (Agudo)
        # Joven, sin historial, pero con sangre/proteína en orina.
        {
            'id': 'PAC-005 (Ana - Agudo)',
            'age': 25, 'bp': 110, 'sg': 1.015, 'al': 2, 'su': 0, 
            'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'notpresent', 'ba': 'present',
            'bgr': 95, 'bu': 25, 'sc': 1.0, 'sod': 142, 'pot': 4.0, 'hemo': 12.0, 
            'pcv': 38, 'wc': 15000, 'rc': 4.8,
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'yes', 'ane': 'no'
        }
    ]

    df = pd.DataFrame(data)
    
    # Guardar en la carpeta data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'data', 'pacientes_nuevos_demo.csv')
    
    df.to_csv(output_path, index=False)
    print(f" Archivo generado: {output_path}")
    print("   -> Sube este archivo a la Web para hacer la Demo honesta.")

if __name__ == "__main__":
    generate_new_patients()