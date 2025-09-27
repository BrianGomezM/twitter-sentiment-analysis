from data_cleaning import clean_data
from data_preprocessing import preprocess
from train import train_model
from evaluate import evaluate_model
import pandas as pd
from utils import mostrar_top_palabras

def main():
    df = clean_data()
    X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer = preprocess(df)
    X = vectorizer.transform(df["text_clean"])

    # Mostrar top palabras de los primeros 10 tweets
    print(mostrar_top_palabras(df, X, vectorizer, top_n=10))

    model, history = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test, encoder)
    import numpy as np



if __name__ == "__main__":
    main()