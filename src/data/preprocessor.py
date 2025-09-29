import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from config import TEXT_CONFIG

class DataPreprocessor:
    """Clase para preparar datos para entrenamiento y prueba (sin validación)"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TEXT_CONFIG['max_features'],
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.encoder = LabelEncoder()
    
    def prepare_data(self, df):
        """
        Prepara datos para entrenamiento y prueba.
        
        Args:
            df (pd.DataFrame): DataFrame con tweets limpios
            
        Returns:
            X_train, X_test, y_train, y_test, encoder, vectorizer
        """
        # Características y etiquetas
        X = df["text_clean"]
        y = df["airline_sentiment"]
        
        # Codificar etiquetas
        y_encoded = self.encoder.fit_transform(y)
        print(f"🔢 Etiquetas codificadas: {dict(zip(self.encoder.classes_, range(len(self.encoder.classes_))))}")
        
        # Vectorizar texto
        print("🔤 Vectorizando texto...")
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"📊 Dimensiones de la matriz TF-IDF: {X_vectorized.shape}")
        
        # Dividir solo en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded,
            test_size=TEXT_CONFIG['test_size'],
            random_state=TEXT_CONFIG['random_state'],
            stratify=y_encoded
        )
        
        print(f"📈 División de datos:")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test, self.encoder, self.vectorizer
