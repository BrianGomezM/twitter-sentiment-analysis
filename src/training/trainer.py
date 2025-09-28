"""
Módulo para entrenamiento del modelo
"""
import time
from models.mlp_model import build_mlp_model
from config import TRAINING_CONFIG

class ModelTrainer:
    """Clase para entrenar el modelo de MLP"""
    
    def __init__(self):
        self.config = TRAINING_CONFIG
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Entrena el modelo MLP con los datos proporcionados
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación
            
        Returns:
            tuple: Modelo entrenado e historial de entrenamiento
        """
        print("🚀 Iniciando entrenamiento del modelo...")
        start_time = time.time()
        
        # Construir modelo
        model = build_mlp_model(X_train.shape[1])
        
        # Entrenar modelo
        history = model.fit(
            X_train.toarray(), y_train,
            validation_data=(X_val.toarray(), y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        return model, history