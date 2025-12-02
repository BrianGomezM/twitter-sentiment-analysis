import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from src.utils import plot_history, evaluate_model

MAX_WORDS = 8000
MAX_LEN = 40

def train_rnn():
    df = pd.read_csv("data/processed_tweets.csv")
    texts = df["clean_standard"].astype(str)
    labels = df["airline_sentiment"]
    encoder = LabelEncoder()
    y = to_categorical(encoder.fit_transform(labels))
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        SimpleRNN(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    history = model.fit(
        X_train, y_train,
        epochs=6,
        batch_size=64,
        validation_split=0.2
    )
    evaluate_model(model, X_test, y_test, encoder, "SimpleRNN")
    plot_history(history, "SimpleRNN")
    return model
