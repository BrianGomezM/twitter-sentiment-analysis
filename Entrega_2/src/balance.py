import pandas as pd
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt

def print_distribution(df, label_col, title="Distribución"):
    counts = Counter(df[label_col])
    total = sum(counts.values())
    for cls, qty in sorted(counts.items()):
        percentage = (qty / total) * 100
        print(f"   • {cls}: {qty:4d} muestras ({percentage:5.1f}%)")
    print(f"   Total: {total} muestras")
    
    return counts

def plot_class_distribution(before_counts, after_counts, title="Balanceo de Clases"):
    classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    before_values = [before_counts.get(cls, 0) for cls in classes]
    bars1 = ax1.bar(range(len(classes)), before_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('ANTES del Balanceo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Clases')
    ax1.set_ylabel('Número de Muestras')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    after_values = [after_counts.get(cls, 0) for cls in classes]
    bars2 = ax2.bar(range(len(classes)), after_values, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax2.set_title('DESPUÉS del Balanceo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Clases')
    ax2.set_ylabel('Número de Muestras')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    before_total = sum(before_values)
    after_total = sum(after_values)
    before_ratios = [v/before_total for v in before_values]
    after_ratios = [v/after_total for v in after_values]
    for i, cls in enumerate(classes):
        print(f"   {cls}: {before_values[i]:4d} → {after_values[i]:4d} "
              f"({before_ratios[i]*100:5.1f}% → {after_ratios[i]*100:5.1f}%)")
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("data/class_balance.png", dpi=150, bbox_inches='tight')
    plt.show()

def smart_balance(df, label_col="airline_sentiment"):
    before_counts = print_distribution(df, label_col, "Distribución ORIGINAL")
    counts_series = df[label_col].value_counts()
    majority_class = counts_series.idxmax()
    majority_count = counts_series.max()
    target_size = int(majority_count * 0.7)
    balanced_dfs = []
    for cls in df[label_col].unique():
        df_cls = df[df[label_col] == cls]
        current_size = len(df_cls)
        if cls == majority_class and current_size > target_size:
            # Undersampling para clase mayoritaria
            df_balanced = df_cls.sample(target_size, random_state=42)
        elif current_size < target_size:
            # Oversampling para clases minoritarias
            df_balanced = resample(df_cls, 
                                  replace=True, 
                                  n_samples=target_size, 
                                  random_state=42)
        else:
            df_balanced = df_cls
        balanced_dfs.append(df_balanced)
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
    after_counts = print_distribution(df_final, label_col, "Distribución BALANCEADA")
    plot_class_distribution(before_counts, after_counts)
    return df_final