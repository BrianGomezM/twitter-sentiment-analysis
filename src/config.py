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

# Parámetros del modelo
MODEL_CONFIG = {
    'hidden_units': [512, 256],
    'dropout_rate': 0.5,
    'activation': 'relu',
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# Parámetros de entrenamiento
TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 32,
    'validation_split': 0.2,
    'test_size': 0.15
}

# Parámetros de preprocesamiento de texto
TEXT_CONFIG = {
    'max_features': 5000,
    'max_len': 100,
    'test_size': 0.2,
    'random_state': 42
}