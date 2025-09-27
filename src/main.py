from data_cleaning import clean_data
from data_preprocessing import preprocess
from train import train_model
from evaluate import evaluate_model
import numpy as np
from utils import plot_history, plot_confusion_matrix


def main():
    df = clean_data()
    X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer = preprocess(df)
    # Mostrar top palabras de los primeros 10 tweets
    #mostrar_top_palabras(df, X_train, vectorizer, n_tweets=10, top_n=10)

    model, history = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test, encoder)
    plot_history(history)

    y_pred = model.predict(X_test.toarray()).argmax(axis=1)
    plot_confusion_matrix(y_test, y_pred, encoder.classes_)

    import numpy as np



if __name__ == "__main__":
    main()