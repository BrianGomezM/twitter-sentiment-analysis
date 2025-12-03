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
    description="API para analizar sentimiento usando modelos LSTM o RNN",
    version="2.0.0"
)

# ------------------------------------
# CONFIGURACIÓN
# ------------------------------------
MAX_LEN = 40
LABELS = ["negative", "neutral", "positive"]

# corregido: usar SOLO modelos/tokenizers que EXISTEN
MODELS = {
    "lstm": {
        "model_path": "models_lstm/best_model_EXP4_REGULARIZADO.keras",
        "tokenizer_path": "models_lstm/tokenizer_EXP3_AVANZADO.json",
        "type": "LSTM Bidirectional"
    },
    "rnn": {
        "model_path": "models_rnn/clean_emb100_rnn64_ep130model.keras",
        "tokenizer_path": "models_rnn/clean_emb100_rnn64_ep130tokenizer.json",
        "type": "Simple RNN"
    }
}

loaded_models = {}
loaded_tokenizers = {}


# ------------------------------------
# UTILIDADES
# ------------------------------------
def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        return tokenizer_from_json(raw)
    except:
        obj = json.loads(raw)
        if isinstance(obj, str):
            obj = json.loads(obj)
        return tokenizer_from_json(json.dumps(obj))


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


# ------------------------------------
# CARGA BAJO DEMANDA
# ------------------------------------
def get_model(model_name):

    if model_name not in MODELS:
        return None, None

    if model_name not in loaded_models:
        print(f"⏳ Loading model {model_name}...")

        mpath = MODELS[model_name]["model_path"]
        tpath = MODELS[model_name]["tokenizer_path"]

        loaded_models[model_name] = load_model(mpath)
        loaded_tokenizers[model_name] = load_tokenizer(tpath)

    return loaded_models[model_name], loaded_tokenizers[model_name]


# ------------------------------------
# REQUEST BODY
# ------------------------------------
class PredictionInput(BaseModel):
    model: str  # "lstm" o "rnn"
    text: str


# ------------------------------------
# ENDPOINT PRINCIPAL
# ------------------------------------
@app.post("/predict")
def predict_sentiment(body: PredictionInput):

    model, tokenizer = get_model(body.model)

    if model is None:
        return {"error": "Modelo inválido. Usa 'lstm' o 'rnn'."}

    clean = clean_text(body.text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(padded)[0].tolist()
    idx = int(np.argmax(probs))
    label = LABELS[idx]

    polarity = probs[2] - probs[0]

    color_map = {
        "positive": "#4CAF50",
        "neutral": "#FFC107",
        "negative": "#F44336"
    }

    emoji_map = {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😠"
    }

    return {
        "input_text": body.text,
        "cleaned_text": clean,
        "sentiment": label,
        "confidence": round(probs[idx], 4),
        "probabilities": {
            "negative": round(probs[0], 4),
            "neutral": round(probs[1], 4),
            "positive": round(probs[2], 4)
        },
        "polarity_index": round(float(polarity), 4),
        "emoji": emoji_map[label],
        "color": color_map[label],
        "model_info": {
            "name": body.model,
            "type": MODELS[body.model]["type"],
            "path": MODELS[body.model]["model_path"]
        }
    }


@app.get("/")
def root():
    return {
        "message": "Sentiment API running 🔥",
        "models_available": list(MODELS.keys())
    }
