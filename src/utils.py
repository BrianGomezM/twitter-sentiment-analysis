import emoji
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def limpiar_texto(texto):
    if texto is None:
        return ""

    # Quitar emojis con la librería emoji
    texto = emoji.replace_emoji(texto, replace=" ")

    # Quitar caracteres no deseados
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.,!?¿¡\s]", " ", texto)

    # Normalizar espacios
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

def mostrar_top_palabras(df, X, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    resultados = []

    # 👇 Tweet número 10 (índice 9 porque empieza en 0)
    i = 9  
    tweet = df.iloc[i]["text_clean"]
    row = X[i].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]

    for idx in top_indices:
        palabra = feature_names[idx]
        peso = row[idx]
        if peso > 0:
            resultados.append({
                "Tweet #": i+1,
                "Texto": tweet,
                "Palabra": palabra,
                "Peso TF-IDF": round(peso, 3)
            })

    tabla = pd.DataFrame(resultados)
    return tabla

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Curvas de entrenamiento ---
def plot_history(history):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


# --- Matriz de confusión ---
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()
