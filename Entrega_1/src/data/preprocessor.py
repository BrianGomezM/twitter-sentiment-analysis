"""
Módulo para preprocesamiento de datos para machine learning - MEJORADO
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from config import get_active_config

class DataPreprocessor:
    """Clase para preparar datos para el modelo de ML - CONFIGURABLE"""
    
    def __init__(self, custom_text_config=None):
        # Usar configuración personalizada o la activa
        if custom_text_config:
            self.text_config = custom_text_config
        else:
            self.text_config = get_active_config()['text']
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.text_config['max_features'],
            stop_words=self.text_config['stop_words'],
            ngram_range=self.text_config['ngram_range'],
            min_df=self.text_config['min_df'],
            max_df=self.text_config['max_df']
        )
        self.encoder = LabelEncoder()
    
    def prepare_data(self, df):
        """
        Prepara los datos para entrenamiento del modelo
        """
        X = df["text_clean"]
        y = df["airline_sentiment"]
        y_encoded = self.encoder.fit_transform(y)
        
        print(f"🔢 Etiquetas codificadas: {dict(zip(self.encoder.classes_, range(len(self.encoder.classes_))))}")
        print("🔤 Vectorizando texto...")
        
        X_vectorized = self.vectorizer.fit_transform(X)
        print(f"📊 Dimensiones de la matriz TF-IDF: {X_vectorized.shape}")
        
        # Usar configuración dinámica
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_vectorized, y_encoded, 
            test_size=self.text_config['test_size'], 
            random_state=self.text_config['random_state'],
            stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.text_config['val_size'],
            random_state=self.text_config['random_state'],
            stratify=y_temp
        )
        
        print(f"📈 División de datos:")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Validación: {X_val.shape[0]} muestras")
        print(f"   - Prueba: {X_test.shape[0]} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.encoder, self.vectorizer

    def get_vectorizer_config(self):
        """Retorna la configuración del vectorizer para referencia"""
        return {
            'max_features': self.text_config['max_features'],
            'ngram_range': self.text_config['ngram_range'],
            'min_df': self.text_config['min_df'],
            'max_df': self.text_config['max_df']
        }