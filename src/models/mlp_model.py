"""
Módulo con la arquitectura del Perceptrón Multicapa (MLP)
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from config import MODEL_CONFIG

def build_mlp_model(input_dim, num_classes=3):
    """
    Construye un modelo MLP para clasificación de texto
    
    Args:
        input_dim (int): Dimensionalidad de las características de entrada
        num_classes (int): Número de clases de salida
        
    Returns:
        tensorflow.keras.Model: Modelo MLP compilado
    """
    print(f"🧠 Construyendo MLP con {input_dim} características de entrada y {num_classes} clases")
    
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Capa oculta 1
        Dense(MODEL_CONFIG['hidden_units'][0], 
              activation=MODEL_CONFIG['activation']),
        Dropout(MODEL_CONFIG['dropout_rate']),
        
        # Capa oculta 2
        Dense(MODEL_CONFIG['hidden_units'][1], 
              activation=MODEL_CONFIG['activation']),
        Dropout(MODEL_CONFIG['dropout_rate']),
        
        # Capa de salida
        Dense(num_classes, activation=MODEL_CONFIG['output_activation'])
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=MODEL_CONFIG['optimizer'],
        loss=MODEL_CONFIG['loss'],
        metrics=MODEL_CONFIG['metrics']
    )
    
    print("✅ Modelo MLP construido y compilado")
    model.summary()
    
    return model