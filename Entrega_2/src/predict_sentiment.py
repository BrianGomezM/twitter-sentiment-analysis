#!/usr/bin/env python3
import os
import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------
MAX_WORDS = 8000
MAX_LEN = 40

LABELS = ["negative", "neutral", "positive"]


# ------------------------------
# CARGAR TOKENIZER (CORREGIDO 100%)
# Acepta tokenizers guardados de 3 formas distintas.
# ------------------------------
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # CASO 1 → JSON válido (modelos LSTM)
    try:
        return tokenizer_from_json(raw)
    except:
        pass

    # CASO 2 → tokenizers RNN guardados con json.dump()
    try:
        # primer decode
        obj = json.loads(raw)
        # si quedó doble codificado, decode otra vez
        if isinstance(obj, str):
            obj = json.loads(obj)
        # convertir a string nuevamente
        return tokenizer_from_json(json.dumps(obj))
    except Exception as e:
        print("❌ Error al decodificar tokenizer:", e)
        print("Archivo:", tokenizer_path)
        sys.exit(1)


# ------------------------------
# LIMPIEZA BÁSICA
# ------------------------------
import re
import emoji
from html import unescape

def clean_for_prediction(text):
    text = unescape(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------
# PREDICCIÓN
# ------------------------------
def predict(model, tokenizer, text):
    cleaned = clean_for_prediction(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(padded)[0]
    idx = np.argmax(probs)
    return LABELS[idx], probs


# ------------------------------
# DIBUJAR BARRA ASCII
# ------------------------------
def draw_bar(label, count, max_count):
    length = int((count / max_count) * 30)
    bar = ":" * length
    print(f"{label:<8} {bar} {count}")


# ------------------------------
# TEST SETS
# ------------------------------
positive_set = [
    "I absolutely loved the service today!",
    "The flight attendants were incredibly kind.",
    "Great experience, I would fly again!",
    "Everything was perfect from start to finish.",
    "Amazing crew, very professional.",
    "Smooth and pleasant flight.",
    "Loved the snacks and entertainment.",
    "Wonderful service, thank you!",
    "This airline never disappoints!",
    "I’m really happy with today’s flight!"
]

neutral_set = [
    "The flight was okay.",
    "I’m waiting to board the plane.",
    "The airport is crowded today.",
    "Just landed at my destination.",
    "My flight leaves in two hours.",
    "Boarding group C is now entering.",
    "I checked in online.",
    "We are cruising at 30,000 feet.",
    "The seatbelt sign is on.",
    "The plane is taxiing to the gate."
]

negative_set = [
    "This is the worst flight ever.",
    "Terrible service today.",
    "My seat was broken the whole flight.",
    "Delays everywhere, horrible experience.",
    "Very rude staff.",
    "Lost my luggage, extremely upset.",
    "The plane was dirty and uncomfortable.",
    "I’m never flying with this airline again.",
    "Worst customer service imaginable.",
    "Everything went wrong today."
]

combined_30 = positive_set[:10] + neutral_set[:10] + negative_set[:10]


# ------------------------------
# PROGRAMA PRINCIPAL
# ------------------------------
def main():
    print("=" * 60)
    print("SENTIMENT ANALYZER – TEST YOUR MODELS")
    print("=" * 60)

    print("\nSeleccione el modelo que desea usar:")
    print("   1. LSTM Bidireccional (models_lstm)")
    print("   2. RNN Simple (models_rnn)")
    print("   3. Ejecutar Test de 30 Frases (auto)")
    choice = input("\nOpción (1/2/3): ").strip()

    # ------------------------------
    # OPCIÓN 3 — AUTO TEST
    # ------------------------------
    if choice == "3":
        model_path = "models_lstm/best_model_EXP4_REGULARIZADO.h5"
        tok_path = "models_lstm/tokenizer_EXP4_REGULARIZADO.json"

        print("\nCargando modelo:", model_path)
        model = load_model(model_path)

        print("Cargando tokenizer:", tok_path)
        tokenizer = load_tokenizer(tok_path)

        print("\nEjecutando test de 30 frases...\n")

        counts = {"positive": 0, "neutral": 0, "negative": 0}

        for text in combined_30:
            label, _ = predict(model, tokenizer, text)
            counts[label] += 1

        max_count = max(counts.values())

        print("\nRESULTADOS:")
        print("-------------------------------")
        draw_bar("Positive", counts["positive"], max_count)
        draw_bar("Neutral", counts["neutral"], max_count)
        draw_bar("Negative", counts["negative"], max_count)
        print("-------------------------------")
        print("\nTest completado ✔")
        return

    # ------------------------------
    # 1️⃣ LSTM
    # ------------------------------
    if choice == "1":
        print("\nModelos en /models_lstm:")
        models = [m for m in os.listdir("models_lstm") if m.endswith(".h5")]

        for i, m in enumerate(models):
            print(f"   {i+1}. {m}")

        pick = int(input("\nElija modelo: ")) - 1
        model_file = models[pick]

        model_path = f"models_lstm/{model_file}"
        tok_name = model_file.replace("best_model_", "").replace(".h5", "")
        tok_path = f"models_lstm/tokenizer_{tok_name}.json"

        model = load_model(model_path)
        tokenizer = load_tokenizer(tok_path)

    # ------------------------------
    # 2️⃣ RNN
    # ------------------------------
    elif choice == "2":
        print("\nModelos en /models_rnn:")
        models = [m for m in os.listdir("models_rnn") if m.endswith("model.keras")]

        for i, m in enumerate(models):
            print(f"   {i+1}. {m}")

        pick = int(input("\nElija modelo: ")) - 1
        model_file = models[pick]

        base = model_file.replace("model.keras", "")
        model_path = f"models_rnn/{model_file}"
        tok_path = f"models_rnn/{base}tokenizer.json"

        model = load_model(model_path)
        tokenizer = load_tokenizer(tok_path)

    else:
        print("Opción inválida")
        return

    # ------------------------------
    # MODO MANUAL
    # ------------------------------
    while True:
        text = input("\nIngrese texto (o 'exit'): ")
        if text.lower() == "exit":
            break

        label, probs = predict(model, tokenizer, text)

        print("\n========================================")
        print(f"Texto: {text}")
        print(f"Predicción: {label.upper()}")
        for i, cls in enumerate(LABELS):
            print(f"   - {cls}: {probs[i]:.4f}")
        print("========================================")


if __name__ == "__main__":
    main()
