# src/train_rnn.py

import os
import json
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils import compute_class_weight, resample
from collections import Counter

MAX_WORDS = 8000
MAX_LEN = 40

from src.utils_rnn import plot_history, evaluate_model


def print_class_distribution(labels, title="Distribución de clases"):
    counts = Counter(labels)
    print(f"\n {title}")
    for c, qty in counts.items():
        print(f"   - {c}: {qty} muestras")
    return counts


def smart_balance_dataset(df, label_col="airline_sentiment"):
    counts = df[label_col].value_counts()
    print("\n Distribución ORIGINAL:")
    for cls, count in counts.items():
        print(f"   - {cls}: {count} muestras")

    min_samples = counts.min()
    max_samples = int(counts.median() * 1.5)

    frames = []

    for cls in counts.index:
        df_cls = df[df[label_col] == cls]

        if len(df_cls) > max_samples:
            df_bal = df_cls.sample(max_samples, random_state=42)
        elif len(df_cls) < min_samples:
            df_bal = resample(df_cls, replace=True,
                              n_samples=min_samples, random_state=42)
        else:
            df_bal = df_cls

        frames.append(df_bal)

    df_final = pd.concat(frames).sample(frac=1, random_state=42)

    print("\n Distribución BALANCEADA:")
    for cls, count in df_final[label_col].value_counts().items():
        print(f"   - {cls}: {count} muestras")

    return df_final



def create_rnn_model(
    embedding_dim,
    rnn_units,
    dense_units,
    max_words=MAX_WORDS,
    max_len=MAX_LEN
):

    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Dropout(0.2),

        SimpleRNN(
            rnn_units,
            activation="tanh",
            dropout=0.2,
            kernel_regularizer=l2(0.003)
        ),

        BatchNormalization(),
        Dropout(0.4),

        Dense(dense_units, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(3, activation="softmax")
    ])

    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model



def train_rnn(
    embedding_dim=64,
    rnn_units=64,
    dense_units=32,
    epochs=15,
    batch_size=32,
    use_class_weights=True
):

    print(f"\n🚀 Entrenando RNN SIN MEMORIA")

    df = pd.read_csv("data/processed_tweets.csv")

    df = smart_balance_dataset(df, "airline_sentiment")

    texts = df["clean_text"].astype(str)
    labels = df["airline_sentiment"]

    print_class_distribution(labels, title="Distribución final para entrenamiento")

    encoder = LabelEncoder()
    y_indices = encoder.fit_transform(labels)
    y = to_categorical(y_indices)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_indices
    )

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


    experiment_dir = "experiments_rnn"
    os.makedirs(experiment_dir, exist_ok=True)

    folder_name = f"clean_emb{embedding_dim}_rnn{rnn_units}_ep{epochs}"
    save_path = os.path.join(experiment_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    model = create_rnn_model(embedding_dim, rnn_units, dense_units)

    print("\n📋 Resumen del modelo:")
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.003,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.5,
        min_lr=0.0003,
        verbose=1
    )

    print("\n🚀 Entrenando RNN sin memoria...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\n📊 Evaluando modelo...")
    evaluate_model(model, X_test, y_test, encoder, f"RNN_{folder_name}", save_path)
    plot_history(history, f"RNN_{folder_name}", save_path)

    model.save(os.path.join(save_path, "model.keras"))
    with open(os.path.join(save_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer.to_json(), f)

    print(f"\n✅ Resultados guardados en: {save_path}")

    return model, history
