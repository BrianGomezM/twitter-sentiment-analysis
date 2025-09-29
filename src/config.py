"""
Configuración global del proyecto Twitter Sentiment Analysis
"""
import os

# Configuración de la base de datos
DB_CONFIG = {
    'host': "dpg-d3bmi5ggjchc738ij1m0-a.oregon-postgres.render.com",
    'user': "redes_neuronales_proyecto_user",
    'password': "qgZaEQHbnkqio5wojYT9VldBH81XYn1k",
    'database': "redes_neuronales_proyecto",
    'port': 5432
}

# Parámetros del modelo - REDUCIR COMPLEJIDAD
MODEL_CONFIG = {
    'hidden_units': [50, 20],  # Reducir capas y neuronas
    'dropout_rate': 0.7,        # Aumentar dropout
    'activation': 'relu',
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# Parámetros de entrenamiento - MÁS ÉPOCAS, LEARNING RATE MÁS BAJO
TRAINING_CONFIG = {
    'epochs': 50,               # Más épocas para suavizar
    'batch_size': 16,           # Batch más grande para más estabilidad
    'validation_split': 0.2,
    'test_size': 0.2,
    'learning_rate': 0.0001     # Learning rate más bajo
}

# Parámetros de preprocesamiento de texto
TEXT_CONFIG = {
    'max_features': 10000,      # Más características para mejor representación
    'max_len': 100,
    'test_size': 0.2,
    'random_state': 42
}