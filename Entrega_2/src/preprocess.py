import pandas as pd
import re
import os
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_PATH = "data/Tweets.csv"

# Descargar recursos nltk si no están instalados
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_minimal(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_standard(text):
    text = clean_minimal(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def clean_aggressive(text):
    text = clean_standard(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 2]
    return " ".join(tokens)

def load_raw_data():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")
        print(f"✔ Datos cargados: {df.shape[0]} filas")
        return df
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def preprocess_data(df):
    df["clean_minimal"] = df["text"].apply(clean_minimal)
    df["clean_standard"] = df["text"].apply(clean_standard)
    df["clean_aggressive"] = df["text"].apply(clean_aggressive)

    return df[[
        "tweet_id",
        "text",
        "clean_minimal",
        "clean_standard",
        "clean_aggressive",
        "airline_sentiment"
    ]]

def save_processed(df):
    output_path = "data/processed_tweets.csv"
    df.to_csv(output_path, index=False)
    print(f"✔ Archivo guardado en {output_path}")

def plot_cleaning_effects(df):
    avg_lengths = {
        "Original": df["text"].astype(str).apply(len).mean(),
        "Minimal": df["clean_minimal"].astype(str).apply(len).mean(),
        "Standard": df["clean_standard"].astype(str).apply(len).mean(),
        "Aggressive": df["clean_aggressive"].astype(str).apply(len).mean(),
    }

    plt.figure(figsize=(10, 5))
    plt.bar(avg_lengths.keys(), avg_lengths.values(),
            color=["#6A5ACD", "#48A9A6", "#F28C28", "#D1495B"])

    plt.title("Comparación de Longitud Promedio por Estrategia de Limpieza")
    plt.xlabel("Estrategia de limpieza")
    plt.ylabel("Longitud promedio (caracteres)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def run_preprocess():
    df = load_raw_data()
    if df is not None:
        df = preprocess_data(df)
        save_processed(df)
        plot_cleaning_effects(df)
