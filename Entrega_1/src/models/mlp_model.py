"""
Módulo con la arquitectura del Perceptrón Multicapa (MLP) - DINÁMICO
Incluye L1+L2, BatchNormalization y Dropout configurable
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from config import get_active_config

def build_mlp(input_dim, num_classes=3, custom_config=None):
    """
    Construye un modelo MLP dinámico para clasificación de texto
    con L1+L2 y BatchNormalization, manteniendo la configuración dinámica.
    """
    if custom_config is None:
        config = get_active_config()
        model_config = config['model']
        training_config = config['training']
    else:
        model_config = custom_config.get('model', get_active_config()['model'])
        training_config = custom_config.get('training', get_active_config()['training'])
    
    hidden_units = model_config['hidden_units']
    dropout_rates = model_config['dropout_rates']
    
    l1_coef = model_config.get('l1', 0.0)
    l2_coef = model_config.get('l2', 0.0)
    
    print(f"🧠 Construyendo MLP DINÁMICO con {input_dim} características")
    print(f"   Arquitectura: {hidden_units}")
    print(f"   Dropout: {dropout_rates}")
    print(f"   Regularización: L1={l1_coef}, L2={l2_coef}")
    
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    # Capas ocultas con L1+L2 y BatchNormalization
    for i, (units, dropout_rate) in enumerate(zip(hidden_units, dropout_rates)):
        model.add(Dense(
            units,
            activation=model_config['activation'],
            kernel_regularizer=l1_l2(l1=l1_coef, l2=l2_coef)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        print(f"   - Capa {i+1}: {units} neuronas, Dropout {dropout_rate}, L1+L2 + BatchNorm")
    
    # Capa de salida
    model.add(Dense(num_classes, activation=model_config['output_activation']))
    
    # Compilar con configuración dinámica
    optimizer = Adam(learning_rate=training_config['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss=training_config['loss'],
        metrics=training_config['metrics']
    )
    
    print("✅ Modelo MLP dinámico construido y compilado")
    return model

def build_mlp_from_params(input_dim, hidden_units, dropout_rates, 
                          activation='relu', output_activation='softmax',
                          l1_coef=0.01, l2_coef=0.01,
                          learning_rate=0.001):
    """
    Construye modelo con parámetros específicos para experimentación
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for units, dropout_rate in zip(hidden_units, dropout_rates):
        model.add(Dense(
            units, activation=activation, kernel_regularizer=l1_l2(l1=l1_coef, l2=l2_coef)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(3, activation=output_activation))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
