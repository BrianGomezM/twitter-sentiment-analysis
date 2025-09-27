from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess(df):

    #Vectorizar texto
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["text_clean"])

    #Extraer etiquetas
    y = df["airline_sentiment"]

    #Etiquetas a números 0, 1, 2
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    #Dividir en train y test (70%, 30%) - stratify: Para que se dividan de forma balanceada
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )
    #Dividir el 30% en val y test (15%, 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer
