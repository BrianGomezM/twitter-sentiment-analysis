"""
Módulo para entrenamiento del modelo - MEJORADO Y DINÁMICO
"""
import time
from models.mlp_model import build_mlp, build_mlp_from_params
from config import CLASS_WEIGHTS, get_active_config
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ModelTrainer:
    """Clase para entrenar el modelo de MLP - MEJORADA Y DINÁMICA"""
    
    def __init__(self, custom_config=None):
        self.custom_config = custom_config
        if custom_config:
            self.config = custom_config['training']
            self.model_config = custom_config['model']
        else:
            active_config = get_active_config()
            self.config = active_config['training']
            self.model_config = active_config['model']
    
    def train_model(self, X_train, y_train, X_val, y_val, custom_model=None):
        """
        Entrena el modelo MLP con configuración dinámica
        """
        print("🚀 Iniciando entrenamiento del modelo DINÁMICO...")
        start_time = time.time()
        
        # Construir modelo personalizado o estándar
        if custom_model:
            model = custom_model
            print("✅ Usando modelo personalizado proporcionado")
        else:
            model = build_mlp(X_train.shape[1], custom_config=self.custom_config)
        
        # Configurar callbacks dinámicamente
        early_stop = EarlyStopping(**self.config['early_stopping'])
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
        
        print(f"⚙️  Configuración de entrenamiento DINÁMICA:")
        print(f"   - Épocas: {self.config['epochs']}")
        print(f"   - Batch size: {self.config['batch_size']}")
        print(f"   - Learning rate: {self.config['learning_rate']}")
        print(f"   - Early stopping: patience={self.config['early_stopping']['patience']}")
        
        history = model.fit(
            X_train.toarray(), y_train,
            validation_data=(X_val.toarray(), y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            class_weight=CLASS_WEIGHTS,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        return model, history
    
    def train_custom_model(self, X_train, y_train, X_val, y_val, 
                          hidden_units, dropout_rates, **kwargs):
        """
        Entrena un modelo con parámetros personalizados para experimentación
        """
        print("🔧 Entrenando modelo con parámetros personalizados...")
        
        model = build_mlp_from_params(
            X_train.shape[1], 
            hidden_units, 
            dropout_rates,
            learning_rate=kwargs.get('learning_rate', 0.001)
        )
        
        return self.train_model(X_train, y_train, X_val, y_val, custom_model=model)