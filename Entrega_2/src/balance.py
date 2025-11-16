# src/balance.py

import pandas as pd
from sklearn.utils import resample
from collections import Counter

def print_distribution(df, label_col, title="Distribución"):
    print(f"\n📊 {title}")
    counts = Counter(df[label_col])
    for c, qty in counts.items():
        print(f"   - {c}: {qty} muestras")
    return counts

def balance_dataset(df, label_col="airline_sentiment", target_size=4000):
    """
    Estrategia híbrida:
    - UNDERSAMPLING para la clase mayoritaria (negative)
    - OVERSAMPLING ligero para las minoritarias (neutral, positive)
    """

    print_distribution(df, label_col, "Distribución ORIGINAL")

    classes = df[label_col].unique()
    frames = []

    for cls in classes:
        df_cls = df[df[label_col] == cls]

        if len(df_cls) > target_size:
            # Reducir negativos
            df_bal = df_cls.sample(target_size, random_state=42)
        else:
            # Aumentar neutrales/positivos sin llegar a 9000
            df_bal = resample(df_cls, replace=True, n_samples=target_size, random_state=42)

        frames.append(df_bal)

    df_final = pd.concat(frames).sample(frac=1, random_state=42)

    print_distribution(df_final, label_col, "Distribución BALANCEADA (Híbrida)")

    return df_final
