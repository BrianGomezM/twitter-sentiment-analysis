"""
Utilidades para visualización de resultados
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """
    Grafica la historia del entrenamiento (pérdida y accuracy)
    
    Args:
        history: Historial de entrenamiento de Keras
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    plt.title('Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
    plt.title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Grafica la matriz de confusión
    
    Args:
        y_true: Etiquetas reales
        y_pred: Etiquetas predichas
        classes: Nombres de las clases
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()