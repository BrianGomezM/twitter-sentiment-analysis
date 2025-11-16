# src/train_lstm.py

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils import plot_history, evaluate_model
from sklearn.utils import compute_class_weight
from sklearn.utils import resample
from collections import Counter

MAX_WORDS = 8000
MAX_LEN = 40
#MAX_WORDS = 20000
#MAX_LEN = 80

def print_class_distribution(labels, title="Distribución de clases"):
    counts = Counter(labels)
    print(f"\n📊 {title}")
    for c, qty in counts.items():
        print(f"   - {c}: {qty} muestras")
    return counts

def smart_balance_dataset(df, label_col="airline_sentiment"):
    """
    Balanceo inteligente: no forzar mismo tamaño para todas las clases
    """
    counts = df[label_col].value_counts()
    print("📊 Distribución ORIGINAL:")
    for cls, count in counts.items():
        print(f"   - {cls}: {count} muestras")
    
    # Estrategia más inteligente: mantener proporciones más naturales
    min_samples = counts.min()
    max_samples = int(counts.median() * 1.5)  # No exagerar el oversampling
    
    frames = []
    for cls in counts.index:
        df_cls = df[df[label_col] == cls]
        current_count = len(df_cls)
        
        if current_count > max_samples:
            # Undersampling suave
            df_bal = df_cls.sample(max_samples, random_state=42)
        elif current_count < min_samples:
            # Oversampling moderado
            df_bal = resample(df_cls, replace=True, 
                            n_samples=min_samples, random_state=42)
        else:
            # Mantener tamaño original
            df_bal = df_cls
            
        frames.append(df_bal)
    
    df_final = pd.concat(frames).sample(frac=1, random_state=42)
    
    print("📊 Distribución BALANCEADA (Inteligente):")
    for cls, count in df_final[label_col].value_counts().items():
        print(f"   - {cls}: {count} muestras")
    
    return df_final

def create_improved_model(embedding_dim, lstm_units, dense_units, max_words=MAX_WORDS, max_len=MAX_LEN):
    """
    Modelo mejorado con regularización
    """
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Dropout(0.3),  # Dropout inicial
        
        LSTM(lstm_units, return_sequences=False,
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        
        BatchNormalization(),
        Dropout(0.4),  # Dropout después de LSTM
        
        Dense(dense_units, activation="relu", 
              kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(3, activation="softmax")
    ])
    
    # Optimizer con learning rate ajustado
    optimizer = Adam(learning_rate=0.0008)  # LR más bajo para mejor estabilidad
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model

def train_lstm(
    cleaning_method="clean_standard",
    embedding_dim=64,
    lstm_units=64,
    dense_units=32,
    epochs=15,  # Más épocas pero con early stopping
    batch_size=32,
    use_class_weights=True
):

    print(f"\n🔶 Entrenando LSTM MEJORADO con limpieza: {cleaning_method}")

    # Cargar datos
    df = pd.read_csv("data/processed_tweets.csv")

    # Balanceo inteligente
    df = smart_balance_dataset(df, "airline_sentiment")

    # Textos y labels
    texts = df[cleaning_method].astype(str)
    labels = df["airline_sentiment"]

    # Mostrar distribución final
    print_class_distribution(labels, title="Distribución para entrenamiento")

    # Codificación de etiquetas
    encoder = LabelEncoder()
    y_indices = encoder.fit_transform(labels)
    y = to_categorical(y_indices)

    # Tokenización
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_indices
    )

    # Calcular class weights
    if use_class_weights:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_indices),
            y=y_indices
        )
        class_weights = {i: float(w) for i, w in enumerate(class_weights)}
        print("\n⚖️ Pesos por clase:", class_weights)
    else:
        class_weights = None

    # Crear directorio para experimentos
    experiment_dir = "experiments_lstm"
    os.makedirs(experiment_dir, exist_ok=True)

    folder_name = f"{cleaning_method}_emb{embedding_dim}_lstm{lstm_units}_ep{epochs}"
    save_path = os.path.join(experiment_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Crear modelo mejorado
    model = create_improved_model(embedding_dim, lstm_units, dense_units)

    # Callbacks para mejor entrenamiento
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )

    print("📋 Resumen del modelo:")
    model.summary()

    # Entrenamiento con callbacks
    print("🚀 Iniciando entrenamiento mejorado...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluación final
    print("📊 Evaluando modelo...")
    evaluate_model(model, X_test, y_test, encoder, f"LSTM_{folder_name}", save_path)
    plot_history(history, f"LSTM_{folder_name}", save_path)
    
    # Guardar modelo y tokenizer
    model.save(os.path.join(save_path, "model.keras"))
    
    with open(os.path.join(save_path, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer.to_json(), f)

    print(f"✅ Resultados guardados en: {save_path}")

    return model, history