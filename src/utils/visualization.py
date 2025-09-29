"""
Utilidades para visualización mejoradas
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    """
    Grafica el historial de entrenamiento con mejoras visuales
    """
    plt.figure(figsize=(15, 6))
    
    # Suavizar curvas
    def smooth_curve(points, factor=0.8):
        smoothed = []
        for point in points:
            if smoothed:
                previous = smoothed[-1]
                smoothed.append(previous * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed
    
    # Accuracy
    plt.subplot(1, 2, 1)
    
    # Curvas suavizadas
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs_range = range(1, len(train_acc) + 1)
    
    plt.plot(epochs_range, train_acc, 'b-', label='Entrenamiento', linewidth=2, alpha=0.8)
    plt.plot(epochs_range, val_acc, 'r-', label='Validación', linewidth=2, alpha=0.8)
    
    plt.title('Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(train_acc))
    
    # Loss
    plt.subplot(1, 2, 2)
    
    train_loss = smooth_curve(history.history['loss'])
    val_loss = smooth_curve(history.history['val_loss'])
    
    plt.plot(epochs_range, train_loss, 'b-', label='Entrenamiento', linewidth=2, alpha=0.8)
    plt.plot(epochs_range, val_loss, 'r-', label='Validación', linewidth=2, alpha=0.8)
    
    plt.title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(train_loss))
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Grafica la matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xlabel('Predicho', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.show()