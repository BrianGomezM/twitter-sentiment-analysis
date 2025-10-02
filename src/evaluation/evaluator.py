"""
Módulo para evaluación del modelo
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils.visualization import plot_confusion_matrix, plot_training_history

class ModelEvaluator:
    """Clase para evaluar el desempeño del modelo"""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, model, X_test, y_test, encoder):
        """
        Evalúa el modelo en el conjunto de prueba
        
        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Etiquetas reales de prueba
            encoder: LabelEncoder usado para codificar etiquetas
        """
        print("Evaluando modelo en conjunto de prueba...")
        
        # Predecir
        y_pred_proba = model.predict(X_test.toarray())
        y_pred = y_pred_proba.argmax(axis=1)
        
        # Reporte de clasificación
        print("\n" + "="*50)
        print("REPORTE DE CLASIFICACIÓN")
        print("="*50)
        report = classification_report(y_test, y_pred, 
                                      target_names=encoder.classes_)
        print(report)
        
        # Matriz de confusión
        plot_confusion_matrix(y_test, y_pred, encoder.classes_)
        
        return y_pred
    
    def get_predictions(self, model, X):
        """
        Obtiene predicciones del modelo
        
        Args:
            model: Modelo entrenado
            X: Datos a predecir
            
        Returns:
            numpy.array: Predicciones
        """
        return model.predict(X.toarray()).argmax(axis=1)