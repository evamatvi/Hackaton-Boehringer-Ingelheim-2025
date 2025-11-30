import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from data_processor import load_and_clean_data

def generate_plots():
    print("Iniciando generación de gráficos clínicos...")
    
    # 1. Cargar datos
    df = load_and_clean_data()
    if df is None:
        return

    # Crear carpeta 'plots' si no existe
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, '..', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Configuración de estilo médico (limpio y profesional)
    sns.set_theme(style="whitegrid")
    
    # --- GRÁFICO 1: Balance de Clases ---
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(x='classification', data=df, palette='viridis')
    plt.title('Distribución de Pacientes (0=Sano, 1=ERC)', fontsize=14)
    plt.xlabel('Diagnóstico')
    plt.ylabel('Número de Pacientes')
    # Añadir etiquetas
    ax.bar_label(ax.containers[0])
    
    save_path = os.path.join(plots_dir, '1_balance_clases.png')
    plt.savefig(save_path)
    print(f"Gráfico 1 guardado: {save_path}")
    plt.close()

    # --- GRÁFICO 2: Creatinina vs Enfermedad (La prueba médica) ---
    plt.figure(figsize=(8, 6))
    # sc = Serum Creatinine
    sns.boxplot(x='classification', y='sc', data=df, palette='Set2')
    plt.title('Niveles de Creatinina según Diagnóstico', fontsize=14)
    plt.xlabel('Diagnóstico (0=Sano, 1=ERC)')
    plt.ylabel('Creatinina Sérica (sc)')
    plt.yscale('log') # Escala logarítmica para ver mejor las diferencias extremas
    
    save_path = os.path.join(plots_dir, '2_creatinina_impacto.png')
    plt.savefig(save_path)
    print(f"Gráfico 2 guardado: {save_path}")
    plt.close()

    # --- GRÁFICO 3: Matriz de Correlación (Top variables) ---
    plt.figure(figsize=(10, 8))
    # Seleccionamos solo las variables más importantes para no saturar
    cols_interes = ['classification', 'history_score', 'sc', 'hemo', 'al', 'htn', 'dm', 'age']
    corr = df[cols_interes].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mapa de Correlaciones Clínicas', fontsize=14)
    
    save_path = os.path.join(plots_dir, '3_correlaciones.png')
    plt.savefig(save_path)
    print(f"Gráfico 3 guardado: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_plots()