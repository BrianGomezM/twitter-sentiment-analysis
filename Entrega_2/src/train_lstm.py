import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from src.utils import plot_enhanced_results, evaluate_model

MAX_WORDS = 8000    
MAX_LEN = 40        

def lr_warmup(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return 0.0001 + (0.001 - 0.0001) * (epoch / warmup_epochs)
    else:
        decay_rate = 0.95
        decay_steps = 5
        if epoch % decay_steps == 0 and epoch > warmup_epochs:
            return lr * decay_rate
    return lr

def create_lstm_model(embedding_dim=64, lstm_units=64, dense_units=32):
    model = Sequential([
        Embedding(input_dim=MAX_WORDS,
                 output_dim=embedding_dim,
                 input_length=MAX_LEN,
                 mask_zero=True,  
                 name="embedding",
                 embeddings_regularizer=l2(0.001)),  
        BatchNormalization(name="batch_norm_embedding"),
        Bidirectional(
            LSTM(units=lstm_units,
                 dropout=0.4,          
                 recurrent_dropout=0.2, 
                 kernel_regularizer=l2(0.001),  
                 recurrent_regularizer=l2(0.001),
                 bias_regularizer=l2(0.001),
                 return_sequences=False,
                 name="lstm"),
            name="bidirectional_lstm"
        ),
        BatchNormalization(name="batch_norm_lstm"),
        Dropout(0.4, name="dropout_1"),
        Dense(units=dense_units,
              activation='relu',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001),
              name="dense_1"),
        BatchNormalization(name="batch_norm_dense1"),
        Dropout(0.3, name="dropout_2"),
        Dense(units=dense_units//2,
              activation='relu',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001),
              name="dense_2"),
        
        BatchNormalization(name="batch_norm_dense2"),
        Dropout(0.2, name="dropout_3"),
        Dense(units=3,
              activation='softmax',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001),
              name="output")
    ])
    
    optimizer = Adam(
        learning_rate=0.001,  
        clipnorm=1.0,        
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def calculate_advanced_metrics(y_true, y_pred, encoder):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    metrics_dict = {}
    for i, class_name in enumerate(encoder.classes_):
        f1 = f1_score(y_true_labels, y_pred_labels, average=None)[i]
        precision = precision_score(y_true_labels, y_pred_labels, average=None)[i]
        recall = recall_score(y_true_labels, y_pred_labels, average=None)[i]
        metrics_dict[f"{class_name}_f1"] = float(f1)
        metrics_dict[f"{class_name}_precision"] = float(precision)
        metrics_dict[f"{class_name}_recall"] = float(recall)
    metrics_dict["macro_f1"] = float(f1_score(y_true_labels, y_pred_labels, average='macro'))
    metrics_dict["macro_precision"] = float(precision_score(y_true_labels, y_pred_labels, average='macro'))
    metrics_dict["macro_recall"] = float(recall_score(y_true_labels, y_pred_labels, average='macro'))
    metrics_dict["weighted_f1"] = float(f1_score(y_true_labels, y_pred_labels, average='weighted'))
    metrics_dict["weighted_precision"] = float(precision_score(y_true_labels, y_pred_labels, average='weighted'))
    metrics_dict["weighted_recall"] = float(recall_score(y_true_labels, y_pred_labels, average='weighted'))
    return metrics_dict

def train_lstm_model(cleaning_method="clean_text",
                    embedding_dim=64,
                    lstm_units=64,
                    dense_units=32,
                    epochs=50,
                    batch_size=32,
                    use_class_weights=True,
                    experiment_name=""):
    df = pd.read_csv("data/balanced_tweets.csv")
    
    texts = df[cleaning_method].astype(str)
    labels = df["airline_sentiment"]
    encoder = LabelEncoder()
    y_indices = encoder.fit_transform(labels)
    y = to_categorical(y_indices)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, 
                     maxlen=MAX_LEN, 
                     padding='post', 
                     truncating='post')
    X_temp, X_test, y_temp, y_test, indices_temp, _ = train_test_split(
        X, y, y_indices,
        test_size=0.20,
        random_state=42,
        stratify=y_indices
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.20,  
        random_state=42,
        stratify=indices_temp
    )
    if use_class_weights:
        print("\nCALCULANDO PESOS DE CLASE...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_indices),
            y=y_indices
        )
        class_weights = {i: float(w) for i, w in enumerate(class_weights)}
        print(f"   • Pesos: {class_weights}")
    else:
        class_weights = None
    model = create_lstm_model(embedding_dim, lstm_units, dense_units)
    model.summary()
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=12,  
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    lr_scheduler = LearningRateScheduler(lr_warmup, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=f'models/best_model_{experiment_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8, 
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [early_stop, lr_scheduler, checkpoint, reduce_lr]
    os.makedirs('models', exist_ok=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    y_test_pred = model.predict(X_test, verbose=0)
    advanced_metrics = calculate_advanced_metrics(y_test, y_test_pred, encoder)
    for class_name in encoder.classes_:
        print(f"\n   📊 Clase: {class_name}")
        print(f"      • F1-Score:    {advanced_metrics[f'{class_name}_f1']:.4f}")
        print(f"      • Precision:   {advanced_metrics[f'{class_name}_precision']:.4f}")
        print(f"      • Recall:      {advanced_metrics[f'{class_name}_recall']:.4f}")
    evaluate_model(model, X_test, y_test, encoder, 
                  f"LSTM_{experiment_name}", 
                  save_path="results")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{experiment_name}_advanced_metrics.json", 'w') as f:
        json.dump(advanced_metrics, f, indent=4)
    plot_enhanced_results(history, f"LSTM_{experiment_name}", save_path="results")
    tokenizer_path = f"models/tokenizer_{experiment_name}.json"
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    return model, history, tokenizer