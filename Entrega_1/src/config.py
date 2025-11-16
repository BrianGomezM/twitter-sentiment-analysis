"""
Configuración global del proyecto Twitter Sentiment Analysis - OPTIMIZADO
Configuración final validada mediante testing exhaustivo
"""

DB_CONFIG = {
    'host': "dpg-d3bmi5ggjchc738ij1m0-a.oregon-postgres.render.com",
    'user': "redes_neuronales_proyecto_user",
    'password': "qgZaEQHbnkqio5wojYT9VldBH81XYn1k",
    'database': "redes_neuronales_proyecto",
    'port': 5432
}

EXPERIMENT_CONFIGS = {
    'optimal': {
        'name': 'Configuración Óptima - Nadam + Batch 48',
        'text': {
            'max_features': 3000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': (1, 1),
            'stop_words': 'english',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42
        },
        'model': {
            'hidden_units': [1024, 512],
            'dropout_rates': [0.7, 0.6],
            'activation': 'relu',
            'output_activation': 'softmax',
            'l1': 0,  
            'l2': 0   
        },
        'training': {
            'epochs': 20,
            'batch_size': 48,  
            'learning_rate': 0.001,
            'optimizer': 'nadam', 
            'early_stopping': {
                'monitor': 'val_loss',
                'patience': 5,
                'restore_best_weights': True
            },
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy']
        }
    }
}

ACTIVE_EXPERIMENT = 'optimal'  

def get_active_config():
    """Retorna la configuración activa actual"""
    return EXPERIMENT_CONFIGS[ACTIVE_EXPERIMENT]

def update_active_config(experiment_name):
    """Actualiza la configuración activa"""
    global ACTIVE_EXPERIMENT
    if experiment_name in EXPERIMENT_CONFIGS:
        ACTIVE_EXPERIMENT = experiment_name
        print(f"✅ Configuración actualizada a: {EXPERIMENT_CONFIGS[experiment_name]['name']}")
    else:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado")

# Configuración actual
current_config = get_active_config()
TEXT_CONFIG = current_config['text']
MODEL_CONFIG = current_config['model']
TRAINING_CONFIG = current_config['training']

# ==================== CONFIGURACIONES FIJAS ====================
CLASS_WEIGHTS = {0: 1.0, 1: 2.5, 2: 1.8}

OPTIMAL_RESULTS = {
    'expected_accuracy': 0.7715,  # Actualizado con resultado real
    'expected_f1_macro': 0.6835,
    'expected_f1_weighted': 0.7567,
    'expected_training_time': 44,
    'optimal_config_note': 'Configuración validada mediante 70+ experimentos - Accuracy: 77.15%'
}

CONFIG_VERSION = "4.0-optimal-nadam-batch48"