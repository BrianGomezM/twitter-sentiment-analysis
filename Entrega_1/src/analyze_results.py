"""
Análisis avanzado de resultados de testing
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def analyze_test_results(results_file):
    """
    Analiza y visualiza resultados de testing
    """
    # Cargar resultados
    df = pd.read_csv(results_file)
    
    print("📊 ANÁLISIS ESTADÍSTICO DE RESULTADOS")
    print("="*60)
    
    # Estadísticas básicas
    print(f"📈 Accuracy promedio: {df['accuracy'].mean():.4f}")
    print(f"🎯 F1 Macro promedio: {df['f1_macro'].mean():.4f}")
    print(f"⏱️  Tiempo promedio: {df['training_time'].mean():.2f}s")
    
    # Top 10 modelos
    top_10 = df.nlargest(10, 'accuracy')[['display_name', 'accuracy', 'f1_macro', 'training_time', 'optimizer']]
    print("\n🏆 TOP 10 MODELOS:")
    print(top_10.to_string(index=False))
    
    # Análisis por categorías
    print("\n🔍 ANÁLISIS POR CATEGORÍAS:")
    
    # Por optimizer
    optimizer_analysis = df.groupby('optimizer').agg({
        'accuracy': ['mean', 'max'],
        'f1_macro': 'mean',
        'training_time': 'mean'
    }).round(4)
    print("\n📊 Por Optimizer:")
    print(optimizer_analysis)
    
    # Por arquitectura (simplificado)
    df['layer_count'] = df['architecture'].apply(lambda x: len(eval(x)))
    layer_analysis = df.groupby('layer_count').agg({
        'accuracy': ['mean', 'max'],
        'training_time': 'mean'
    }).round(4)
    print("\n🏗️  Por Número de Capas:")
    print(layer_analysis)
    
    # Visualizaciones
    plt.figure(figsize=(15, 10))
    
    # 1. Accuracy vs Tiempo
    plt.subplot(2, 2, 1)
    plt.scatter(df['training_time'], df['accuracy'], alpha=0.6)
    plt.xlabel('Tiempo de Entrenamiento (s)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Tiempo de Entrenamiento')
    
    # 2. Top modelos por accuracy
    plt.subplot(2, 2, 2)
    top_10_plot = df.nlargest(10, 'accuracy')
    plt.barh(range(10), top_10_plot['accuracy'])
    plt.yticks(range(10), top_10_plot['display_name'], fontsize=8)
    plt.xlabel('Accuracy')
    plt.title('Top 10 Modelos por Accuracy')
    
    # 3. Distribución de accuracy
    plt.subplot(2, 2, 3)
    plt.hist(df['accuracy'], bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Accuracy')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Accuracy')
    
    # 4. Accuracy por optimizer
    plt.subplot(2, 2, 4)
    df.boxplot(column='accuracy', by='optimizer', ax=plt.gca())
    plt.title('Accuracy por Optimizer')
    plt.suptitle('')  # Eliminar título automático
    
    plt.tight_layout()
    plt.savefig(f'analysis_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Ejemplo de uso - reemplaza con tu archivo de resultados
    analyze_test_results("advanced_test_results_20250928_200614.csv")