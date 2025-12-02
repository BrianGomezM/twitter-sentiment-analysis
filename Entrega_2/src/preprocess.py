"""
Módulo de preprocesamiento de tweets
Limpieza optimizada conservando información emocional
"""
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import emoji
from html import unescape

DATA_PATH = "data/Tweets.csv"

def clean_twitter_optimized(text):

    if not isinstance(text, str):
        text = str(text)
    # 1. Convertir entidades HTML (&amp; -> &, &lt; -> <, etc.)
    text = unescape(text)
    # 2. Convertir emojis a texto descriptivo
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 3. Eliminar URLs
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    # 4. Normalizar menciones de usuario
    text = re.sub(r"@\w+", "@usuario", text)
    # 5. Hashtags: remover # pero conservar la palabra
    text = re.sub(r"#(\w+)", r"\1", text)
    # 6. Normalizar repeticiones de signos de puntuación emocional
    text = re.sub(r"([!?])\1+", r"\1", text)  # !! -> !
    text = re.sub(r"\.{2,}", "...", text)     # ... -> ... (normalizado)
    # 7. Eliminar números (no aportan para análisis de sentimiento)
    text = re.sub(r"\d+", "", text)
    # 8. Convertir a minúsculas
    text = text.lower()
    # 9. Limpiar espacios extra
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def load_raw_data():
    """Cargar datos crudos desde CSV"""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")
        return df
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def preprocess_data(df):
    df["clean_text"] = df["text"].apply(clean_twitter_optimized)
    # Mantener solo columnas necesarias
    result_df = df[["tweet_id", "text", "clean_text", "airline_sentiment"]]
    # Mostrar ejemplos
    for i in range(min(3, len(result_df))):
        print(f"Original: {result_df.iloc[i]['text'][:80]}...")
        print(f"Limpio:   {result_df.iloc[i]['clean_text'][:80]}...")
        print("-" * 40)
    
    return result_df

def save_processed(df):
    """Guardar datos procesados"""
    output_path = "data/processed_tweets.csv"
    df.to_csv(output_path, index=False)

def plot_cleaning_effects(df):
    """Visualizar efecto de la limpieza"""
    avg_lengths = {
        "Original": df["text"].astype(str).apply(len).mean(),
        "Limpio": df["clean_text"].astype(str).apply(len).mean(),
    }
    
    vocab_reduction = (1 - avg_lengths["Limpio"] / avg_lengths["Original"]) * 100

    plt.figure(figsize=(12, 5))
    
    # Gráfica de barras
    plt.subplot(1, 2, 1)
    bars = plt.bar(avg_lengths.keys(), avg_lengths.values(),
                   color=["#FF6B6B", "#4ECDC4"], alpha=0.8)
    plt.title("Longitud Promedio de Textos", fontsize=14, fontweight='bold')
    plt.xlabel("Tipo de texto")
    plt.ylabel("Caracteres promedio")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Añadir valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica de porcentaje de reducción
    plt.subplot(1, 2, 2)
    plt.bar(["Reducción"], [vocab_reduction], color="#45B7D1", alpha=0.8)
    plt.title("Reducción de Vocabulario", fontsize=14, fontweight='bold')
    plt.ylabel("Porcentaje (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.text(0, vocab_reduction + 2, f'{vocab_reduction:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.suptitle("Efecto del Preprocesamiento en los Datos", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("data/cleaning_effects.png", dpi=150, bbox_inches='tight')
    plt.show()

def run_preprocess():
    """Ejecutar todo el pipeline de preprocesamiento"""
    print("🚀 INICIANDO PREPROCESAMIENTO")
    print("-" * 40)
    
    df = load_raw_data()
    if df is not None:
        df = preprocess_data(df)
        save_processed(df)
        plot_cleaning_effects(df)
        print("Preprocesamiento completado exitosamente")
