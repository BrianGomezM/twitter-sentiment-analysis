"""
Módulo de balanceo inteligente de clases
Estrategia híbrida: undersampling + oversampling
"""
import pandas as pd
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt

def print_distribution(df, label_col, title="Distribución"):
    """Imprimir distribución de clases"""
    counts = Counter(df[label_col])
    total = sum(counts.values())
    
    print(f"\n📊 {title}:")
    print("-" * 30)
    for cls, qty in sorted(counts.items()):
        percentage = (qty / total) * 100
        print(f"   • {cls}: {qty:4d} muestras ({percentage:5.1f}%)")
    print(f"   Total: {total} muestras")
    
    return counts

def plot_class_distribution(before_counts, after_counts, title="Balanceo de Clases"):
    """Visualizar distribución antes y después del balanceo"""
    classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Antes del balanceo
    before_values = [before_counts.get(cls, 0) for cls in classes]
    bars1 = ax1.bar(range(len(classes)), before_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('ANTES del Balanceo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Clases')
    ax1.set_ylabel('Número de Muestras')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes)
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir valores encima de barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Después del balanceo
    after_values = [after_counts.get(cls, 0) for cls in classes]
    bars2 = ax2.bar(range(len(classes)), after_values, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('DESPUÉS del Balanceo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Clases')
    ax2.set_ylabel('Número de Muestras')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.grid(axis='y', alpha=0.3)
    
    # Añadir valores encima de barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Calcular y mostrar estadísticas
    before_total = sum(before_values)
    after_total = sum(after_values)
    before_ratios = [v/before_total for v in before_values]
    after_ratios = [v/after_total for v in after_values]
    
    print("\n📈 ESTADÍSTICAS DE BALANCEO:")
    print("-" * 40)
    for i, cls in enumerate(classes):
        print(f"   {cls}: {before_values[i]:4d} → {after_values[i]:4d} "
              f"({before_ratios[i]*100:5.1f}% → {after_ratios[i]*100:5.1f}%)")
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("data/class_balance.png", dpi=150, bbox_inches='tight')
    plt.show()

def smart_balance(df, label_col="airline_sentiment"):
    """
    Balanceo inteligente usando estrategia híbrida
    
    Args:
        df: DataFrame con los datos
        label_col: Nombre de la columna con las etiquetas
        
    Returns:
        DataFrame balanceado
    """
    print("🎯 APLICANDO BALANCEO INTELIGENTE")
    print("-" * 40)
    
    # Distribución original
    before_counts = print_distribution(df, label_col, "Distribución ORIGINAL")
    
    # Análisis de distribución
    counts_series = df[label_col].value_counts()
    majority_class = counts_series.idxmax()
    majority_count = counts_series.max()
    
    # Estrategia: usar ~70% de la clase mayoritaria como objetivo
    target_size = int(majority_count * 0.7)
    print(f"\n⚙️  Parámetros de balanceo:")
    print(f"   • Clase mayoritaria: {majority_class} ({majority_count} muestras)")
    print(f"   • Tamaño objetivo: {target_size} muestras por clase")
    print(f"   • Estrategia: {'Undersampling' if majority_count > target_size else 'Oversampling'}")
    
    balanced_dfs = []
    
    for cls in df[label_col].unique():
        df_cls = df[df[label_col] == cls]
        current_size = len(df_cls)
        
        print(f"\n   Procesando clase '{cls}':")
        print(f"     • Muestras actuales: {current_size}")
        
        if cls == majority_class and current_size > target_size:
            # Undersampling para clase mayoritaria
            df_balanced = df_cls.sample(target_size, random_state=42)
            print(f"     • Aplicando UNDERSAMPLING → {target_size} muestras")
        elif current_size < target_size:
            # Oversampling para clases minoritarias
            df_balanced = resample(df_cls, 
                                  replace=True, 
                                  n_samples=target_size, 
                                  random_state=42)
            print(f"     • Aplicando OVERSAMPLING → {target_size} muestras")
        else:
            # Mantener tamaño actual si ya está balanceado
            df_balanced = df_cls
            print(f"     • Manteniendo tamaño actual")
        
        balanced_dfs.append(df_balanced)
    
    # Combinar todas las clases
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
    
    # Distribución final
    after_counts = print_distribution(df_final, label_col, "Distribución BALANCEADA")
    
    # Visualizar
    plot_class_distribution(before_counts, after_counts)
    
    print("\n✅ Balanceo completado exitosamente")
    return df_final