from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import re
import emoji
from html import unescape

app = FastAPI(
    title="Sentiment Analysis API",
    description="API para analizar sentimiento usando modelos LSTM y RNN",
    version="1.0.0"
)


# ------------------------------
# CONFIG GENERAL
# ------------------------------
MAX_LEN = 40
LABELS = ["negative", "neutral", "positive"]


# ------------------------------
# REPARACIÓN AUTOMÁTICA DEL TOKENIZER
# ------------------------------
def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Caso 1: JSON correcto
    try:
        return tokenizer_from_json(raw)
    except:
        pass

    # Caso 2: JSON mal guardado (doble codificación)
    obj = json.loads(raw)
    if isinstance(obj, str):
        obj = json.loads(obj)

    return tokenizer_from_json(json.dumps(obj))


# ------------------------------
# LIMPIEZA IGUAL A TU MODELO
# ------------------------------
def clean_text(text):
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
# CARGAR MODELOS (solo 1 por defecto)
# ------------------------------
MODEL_PATH = "models_lstm/best_model_EXP4_REGULARIZADO.h5"
TOKENIZER_PATH = "models_lstm/tokenizer_EXP4_REGULARIZADO.json"

model = load_model(MODEL_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)



# ------------------------------
# INPUT DE LA API
# ------------------------------
class PredictionInput(BaseModel):
    text: str



# ------------------------------
# ENDPOINT PRINCIPAL
# ------------------------------
@app.post("/predict")
def predict_sentiment(item: PredictionInput):
    clean = clean_text(item.text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(padded)[0].tolist()
    idx = int(np.argmax(probs))
    sentiment = LABELS[idx]

    return {
        "input_text": item.text,
        "cleaned": clean,
        "sentiment": sentiment,
        "probabilities": {
            "negative": probs[0],
            "neutral": probs[1],
            "positive": probs[2]
        },
        "model_used": MODEL_PATH
    }



# ------------------------------
# ENDPOINT DE PRUEBA
# ------------------------------
@app.get("/")
def root():
    return {"message": "API is running ✔ Sentiment Analyzer Ready 🚀"}
