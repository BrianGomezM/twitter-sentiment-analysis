# src/utils.py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import numpy as np
import json

def plot_history(history, model_name, save_path=None):
    """Gráficas mejoradas con más métricas y estilo profesional"""
    plt.figure(figsize=(12, 5))

    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train", linewidth=2, marker='o', markersize=4)
    plt.plot(history.history["val_loss"], label="Validation", linewidth=2, marker='s', markersize=4)
    plt.title(f"{model_name} - Loss", fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfica de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train", linewidth=2, marker='o', markersize=4)
    plt.plot(history.history["val_accuracy"], label="Validation", linewidth=2, marker='s', markersize=4)
    plt.title(f"{model_name} - Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "loss_accuracy.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

    # Guardar métricas para análisis
    if save_path:
        metrics = {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'overfitting_gap': float(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])
        }
        
        with open(os.path.join(save_path, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

def evaluate_model(model, X_test, y_test, encoder, model_name, save_path=None):
    """Evaluación mejorada con más métricas"""
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    report = classification_report(y_true, y_pred_labels, target_names=encoder.classes_, digits=4)
    print(report)

    # Matriz de confusión mejorada
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                cbar_kws={'shrink': 0.8})
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        with open(os.path.join(save_path, "classification_report.txt"), "w") as file:
            file.write(report)
    
    plt.close()

    return report