"""
Módulo de entrenamiento LSTM con mejoras solicitadas:
- Dropout en LSTM (stopping dropout)
- Máscara para padding
- Early Stopping
- Split 80/20 entrenamiento/validación
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.utils import plot_enhanced_results, evaluate_model

# Hiperparámetros
MAX_WORDS = 8000    # Tamaño del vocabulario
MAX_LEN = 40        # Longitud máxima de secuencia

def create_lstm_model(embedding_dim=64, lstm_units=64, dense_units=32):
    """
    Crear modelo LSTM con las mejoras solicitadas
    
    Args:
        embedding_dim: Dimensión del embedding
        lstm_units: Número de unidades LSTM
        dense_units: Número de unidades en capas densas
        
    Returns:
        Modelo Keras compilado
    """
    print(f"   • Embedding dim: {embedding_dim}")
    print(f"   • LSTM units: {lstm_units}")
    print(f"   • Dense units: {dense_units}")
    print(f"   • Dropout LSTM: 0.3 (input y recurrent)")
    
    model = Sequential([
        # Capa de Embedding con máscara automática para padding
        Embedding(input_dim=MAX_WORDS,
                 output_dim=embedding_dim,
                 input_length=MAX_LEN,
                 mask_zero=True,  # Máscara para zeros de padding
                 name="embedding"),
        
        # Capa LSTM con dropout (stopping dropout)
        LSTM(units=lstm_units,
             dropout=0.3,           # Dropout para inputs
             recurrent_dropout=0.3, # Dropout para estados recurrentes
             kernel_regularizer=l2(0.001),  # Regularización L2
             bias_regularizer=l2(0.001),
             return_sequences=False,
             name="lstm"),
        
        # Batch Normalization para estabilizar entrenamiento
        BatchNormalization(name="batch_norm_1"),
        
        # Dropout adicional para prevenir overfitting
        Dropout(0.5, name="dropout_1"),
        
        # Capa densa con regularización
        Dense(units=dense_units,
              activation='relu',
              kernel_regularizer=l2(0.001),
              name="dense_1"),
        
        BatchNormalization(name="batch_norm_2"),
        Dropout(0.4, name="dropout_2"),
        
        # Capa densa adicional
        Dense(units=dense_units//2,
              activation='relu',
              kernel_regularizer=l2(0.001),
              name="dense_2"),
        
        BatchNormalization(name="batch_norm_3"),
        Dropout(0.3, name="dropout_3"),
        
        # Capa de salida para 3 clases
        Dense(units=3,
              activation='softmax',
              name="output")
    ])
    
    # Optimizador Adam con learning rate configurable
    optimizer = Adam(learning_rate=0.0008,
                    clipnorm=1.0)  # Evitar exploding gradients
    
    # Compilar modelo
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_lstm_model(cleaning_method="clean_text",
                    embedding_dim=64,
                    lstm_units=64,
                    dense_units=32,
                    epochs=50,
                    batch_size=32,
                    use_class_weights=True,
                    experiment_name=""):
    """
    Entrenar modelo LSTM con split 80/20
    
    Args:
        cleaning_method: Método de limpieza usado
        embedding_dim: Dimensión del embedding
        lstm_units: Unidades LSTM
        dense_units: Unidades en capas densas
        epochs: Número máximo de épocas
        batch_size: Tamaño del batch
        use_class_weights: Usar pesos de clase para imbalance
        experiment_name: Nombre del experimento
        
    Returns:
        model, history, tokenizer
    """
    print(f"\n{'='*60}")
    print(f"🔬 EXPERIMENTO LSTM: {experiment_name}")
    print(f"{'='*60}")
    print(f"📋 CONFIGURACIÓN:")
    print(f"   • Método limpieza: {cleaning_method}")
    print(f"   • Split: 80% entrenamiento / 20% validación")
    print(f"   • Máx épocas: {epochs} (con early stopping)")
    print(f"   • Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # ========== CARGA DE DATOS ==========
    print("📂 CARGANDO DATOS BALANCEADOS...")
    df = pd.read_csv("data/balanced_tweets.csv")
    
    texts = df[cleaning_method].astype(str)
    labels = df["airline_sentiment"]
    
    print(f"   • Total muestras: {len(df)}")
    print(f"   • Columnas: {list(df.columns)}")
    
    # ========== PREPARACIÓN DE DATOS ==========
    print("\n🔧 PREPARANDO DATOS...")
    
    # Codificar etiquetas
    encoder = LabelEncoder()
    y_indices = encoder.fit_transform(labels)
    y = to_categorical(y_indices)
    
    print(f"   • Clases: {encoder.classes_}")
    print(f"   • Distribución: {np.bincount(y_indices)}")
    
    # Tokenización
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, 
                     maxlen=MAX_LEN, 
                     padding='post', 
                     truncating='post')
    
    print(f"   • Vocabulario: {len(tokenizer.word_index)} palabras")
    print(f"   • Secuencias padding: {X.shape}")
    
    # ========== SPLIT 80/20 ==========
    print("\n📊 DIVIDIENDO DATOS (80/20)...")
    
    # Separar test (20%)
    X_temp, X_test, y_temp, y_test, indices_temp, _ = train_test_split(
        X, y, y_indices,
        test_size=0.20,
        random_state=42,
        stratify=y_indices
    )
    
    # De lo restante, 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.20,  # 20% de X_temp
        random_state=42,
        stratify=indices_temp
    )
    
    print(f"   ✓ Train:      {X_train.shape[0]:5d} muestras ({X_train.shape[0]/len(X)*100:5.1f}%)")
    print(f"   ✓ Validation: {X_val.shape[0]:5d} muestras ({X_val.shape[0]/len(X)*100:5.1f}%)")
    print(f"   ✓ Test:       {X_test.shape[0]:5d} muestras ({X_test.shape[0]/len(X)*100:5.1f}%)")
    
    # ========== PESOS DE CLASE ==========
    if use_class_weights:
        print("\n⚖️  CALCULANDO PESOS DE CLASE...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_indices),
            y=y_indices
        )
        class_weights = {i: float(w) for i, w in enumerate(class_weights)}
        print(f"   • Pesos: {class_weights}")
    else:
        class_weights = None
    
    # ========== CREACIÓN DEL MODELO ==========
    print("\n🏗️  CONSTRUYENDO MODELO LSTM...")
    model = create_lstm_model(embedding_dim, lstm_units, dense_units)
    
    # Resumen del modelo
    model.summary()
    
    # ========== CALLBACKS ==========
    print("\n⏱️  CONFIGURANDO CALLBACKS...")
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    # Reduce Learning Rate on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(
        filepath=f'models/best_model_{experiment_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stop, reduce_lr, checkpoint]
    
    # Crear directorio para modelos
    os.makedirs('models', exist_ok=True)
    
    # ========== ENTRENAMIENTO ==========
    print("\n🎯 INICIANDO ENTRENAMIENTO...")
    print("-" * 50)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========== EVALUACIÓN ==========
    print("\n📈 EVALUANDO MODELO EN TEST...")
    evaluate_model(model, X_test, y_test, encoder, 
                  f"LSTM_{experiment_name}", 
                  save_path="results")
    
    # ========== GRÁFICAS ==========
    print("\n📊 GENERANDO GRÁFICAS DE ANÁLISIS...")
    os.makedirs("results", exist_ok=True)
    plot_enhanced_results(history, f"LSTM_{experiment_name}", save_path="results")
    
    # ========== GUARDAR TOKENIZER ==========
    tokenizer_path = f"models/tokenizer_{experiment_name}.json"
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    
    print(f"\n✅ EXPERIMENTO {experiment_name} COMPLETADO")
    print(f"   • Modelo guardado: models/best_model_{experiment_name}.h5")
    print(f"   • Tokenizer guardado: {tokenizer_path}")
    print(f"   • Resultados en: results/")
    
    return model, history, tokenizer