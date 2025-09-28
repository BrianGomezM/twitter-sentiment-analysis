"""
Módulo para preprocesamiento de datos para machine learning
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from config import TEXT_CONFIG

class DataPreprocessor:
    """Clase para preparar datos para el modelo de ML"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TEXT_CONFIG['max_features'],
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.encoder = LabelEncoder()
    
    def prepare_data(self, df):
        """
        Prepara los datos para entrenamiento del modelo
        Args:
            df (pandas.DataFrame): DataFrame con tweets limpios
            
        Returns:
            tuple: Datos divididos y objetos de preprocesamiento
        """
        # 1. Preparar características (X) y etiquetas (y)
        X = df["text_clean"]
        y = df["airline_sentiment"]
        
        # 2. Codificar etiquetas (negative, neutral, positive -> 0, 1, 2)
        y_encoded = self.encoder.fit_transform(y)
        print(f"🔢 Etiquetas codificadas: {dict(zip(self.encoder.classes_, range(len(self.encoder.classes_))))}")
        
        # 3. Vectorizar texto (TF-IDF)
        print("🔤 Vectorizando texto...")
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"📊 Dimensiones de la matriz TF-IDF: {X_vectorized.shape}")
        
        # 4. Dividir datos (Train/Validation/Test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_vectorized, y_encoded, 
            test_size=TEXT_CONFIG['test_size'], 
            random_state=TEXT_CONFIG['random_state'],
            stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.2,  # 20% of remaining for validation
            random_state=TEXT_CONFIG['random_state'],
            stratify=y_temp
        )
        
        print(f"📈 División de datos:")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Validación: {X_val.shape[0]} muestras")
        print(f"   - Prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.encoder, self.vectorizer