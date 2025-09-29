"""
Módulo para entrenamiento del modelo
"""
import time
from models.mlp_model import build_mlp_model
from config import TRAINING_CONFIG
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
class ModelTrainer:
    """Clase para entrenar el modelo de MLP"""
    
    def __init__(self):
        self.config = TRAINING_CONFIG
    
    def train_model(self, X_train, y_train):
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
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        # Entrenar modelo
        history = model.fit(
            X_train.toarray(), y_train,
            validation_split=0.2,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")

# 🔥 Resultados finales
        final_loss = history.history["loss"][-1]
        final_acc = history.history["accuracy"][-1]
        final_val_loss = history.history["val_loss"][-1]
        final_val_acc = history.history["val_accuracy"][-1]

        print("\n📊 RESULTADOS FINALES DEL MODELO")
        print("="*40)
        print(f"✅ Loss de entrenamiento: {final_loss:.4f}")
        print(f"✅ Accuracy de entrenamiento: {final_acc:.4f}")
        print(f"✅ Loss de validación: {final_val_loss:.4f}")
        print(f"✅ Accuracy de validación: {final_val_acc:.4f}")
        return model, history