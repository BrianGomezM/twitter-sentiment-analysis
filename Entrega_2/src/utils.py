"""
Módulo de utilidades: gráficas y evaluación
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import numpy as np
import json

def plot_enhanced_results(history, model_name, save_path=None):
    """
    Gráficas profesionales para análisis del entrenamiento
    
    Args:
        history: Historial de entrenamiento de Keras
        model_name: Nombre del modelo para el título
        save_path: Ruta para guardar las gráficas (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ========== GRÁFICA 1: PÉRDIDA ==========
    axes[0, 0].plot(history.history['loss'], 
                   label='Entrenamiento', 
                   linewidth=2.5, 
                   color='blue', 
                   marker='o', 
                   markersize=4)
    axes[0, 0].plot(history.history['val_loss'], 
                   label='Validación', 
                   linewidth=2.5, 
                   color='red', 
                   marker='s', 
                   markersize=4)
    axes[0, 0].set_title(f'{model_name} - Evolución de la Pérdida', 
                        fontsize=14, 
                        fontweight='bold')
    axes[0, 0].set_xlabel('Épocas')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].fill_between(range(len(history.history['loss'])), 
                           history.history['loss'], 
                           history.history['val_loss'], 
                           alpha=0.1, 
                           color='gray')
    
    # ========== GRÁFICA 2: PRECISIÓN ==========
    axes[0, 1].plot(history.history['accuracy'], 
                   label='Entrenamiento', 
                   linewidth=2.5, 
                   color='green', 
                   marker='o', 
                   markersize=4)
    axes[0, 1].plot(history.history['val_accuracy'], 
                   label='Validación', 
                   linewidth=2.5, 
                   color='orange', 
                   marker='s', 
                   markersize=4)
    axes[0, 1].set_title(f'{model_name} - Evolución de la Precisión', 
                        fontsize=14, 
                        fontweight='bold')
    axes[0, 1].set_xlabel('Épocas')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # ========== GRÁFICA 3: BRECA DE OVERFITTING ==========
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    overfitting_gap = train_acc - val_acc
    
    axes[1, 0].plot(overfitting_gap, 
                   linewidth=2.5, 
                   color='purple', 
                   marker='^', 
                   markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].fill_between(range(len(overfitting_gap)), 
                           overfitting_gap, 
                           alpha=0.3, 
                           color='purple')
    axes[1, 0].set_title('Brecha de Overfitting (Train - Val)', 
                        fontsize=14, 
                        fontweight='bold')
    axes[1, 0].set_xlabel('Épocas')
    axes[1, 0].set_ylabel('Diferencia de Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Anotar brecha máxima
    max_gap_idx = np.argmax(overfitting_gap)
    max_gap = overfitting_gap[max_gap_idx]
    axes[1, 0].annotate(f'Máx: {max_gap:.4f}',
                       xy=(max_gap_idx, max_gap),
                       xytext=(max_gap_idx, max_gap + 0.02),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, 
                       color='red')
    
    # ========== GRÁFICA 4: LEARNING RATE ==========
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], 
                       linewidth=2.5, 
                       color='brown', 
                       marker='D', 
                       markersize=4)
        axes[1, 1].set_title('Schedule del Learning Rate', 
                            fontsize=14, 
                            fontweight='bold')
        axes[1, 1].set_xlabel('Épocas')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 
                       'Learning Rate Schedule\nno disponible\n(usando LR constante)', 
                       ha='center', 
                       va='center', 
                       fontsize=12)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
    
    plt.suptitle(f'Análisis Detallado: {model_name}', 
                fontsize=16, 
                fontweight='bold', 
                y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{model_name}_analysis.png"), 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white')
    
    plt.show()
    
    # ========== GUARDAR MÉTRICAS ==========
    if save_path:
        metrics = {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'overfitting_gap': float(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['loss']),
            'best_val_acc': float(np.max(history.history['val_accuracy'])),
            'best_val_loss': float(np.min(history.history['val_loss'])),
            'best_epoch': int(np.argmin(history.history['val_loss']) + 1)
        }
        
        with open(os.path.join(save_path, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

def evaluate_model(model, X_test, y_test, encoder, model_name, save_path=None):
    """
    Evaluar modelo y generar reportes
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        encoder: LabelEncoder usado
        model_name: Nombre del modelo
        save_path: Ruta para guardar resultados
    """
    print(f"\n📊 EVALUANDO MODELO: {model_name}")
    print("-" * 50)
    
    # Predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    # Reporte de clasificación
    report = classification_report(y_true, y_pred_labels, 
                                   target_names=encoder.classes_, 
                                   digits=4)
    print("📈 REPORTE DE CLASIFICACIÓN:")
    print(report)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                cbar_kws={'shrink': 0.8},
                annot_kws={"size": 12})
    plt.title(f"Matriz de Confusión - {model_name}", 
              fontsize=16, 
              fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{model_name}_confusion.png"), 
                   dpi=300, 
                   bbox_inches='tight')
        with open(os.path.join(save_path, f"{model_name}_report.txt"), "w") as file:
            file.write(f"Model: {model_name}\n")
            file.write("=" * 50 + "\n")
            file.write(report)
    
    plt.show()
    
    # Calcular accuracy por clase
    class_accuracies = {}
    for i, cls in enumerate(encoder.classes_):
        idx = (y_true == i)
        if sum(idx) > 0:
            accuracy = np.mean(y_pred_labels[idx] == y_true[idx])
            class_accuracies[cls] = accuracy
    
    print("\n🎯 ACCURACY POR CLASE:")
    for cls, acc in class_accuracies.items():
        print(f"   • {cls}: {acc:.4f}")
    
    return report