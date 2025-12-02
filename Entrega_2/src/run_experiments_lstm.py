import os
import pandas as pd
from src.train_lstm import train_lstm_model

def run_all_experiments():
    # Crear directorios para resultados
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Método de limpieza (solo uno, el optimizado)
    cleaning_methods = ["clean_text"]
    
    # Configuraciones de hiperparámetros
    experiments = [
        {
            "name": "EXP1_BASICO",
            "description": "Configuración básica con dropout 0.3",
            "params": {
                "embedding_dim": 50,
                "lstm_units": 32,
                "dense_units": 16,
                "epochs": 50,
                "batch_size": 32
            }
        },
        {
            "name": "EXP2_BALANCEADO", 
            "description": "Configuración balanceada recomendada",
            "params": {
                "embedding_dim": 64,
                "lstm_units": 64,
                "dense_units": 32,
                "epochs": 50,
                "batch_size": 32
            }
        },
        {
            "name": "EXP3_AVANZADO",
            "description": "Configuración avanzada con más capacidad",
            "params": {
                "embedding_dim": 100,
                "lstm_units": 128,
                "dense_units": 64,
                "epochs": 50,
                "batch_size": 64
            }
        },
        {
            "name": "EXP4_REGULARIZADO",
            "description": "Configuración con más regularización",
            "params": {
                "embedding_dim": 64,
                "lstm_units": 96,
                "dense_units": 48,
                "epochs": 50,
                "batch_size": 32
            }
        }
    ]
    
    results_summary = []
    
    for cleaning in cleaning_methods:
        for exp in experiments:
            exp_name = exp["name"]
            params = exp["params"] 
            try:
                # Entrenar modelo
                model, history, tokenizer = train_lstm_model(
                    cleaning_method=cleaning,
                    embedding_dim=params["embedding_dim"],
                    lstm_units=params["lstm_units"],
                    dense_units=params["dense_units"],
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    experiment_name=exp_name
                )
                
                # Recopilar resultados
                final_epoch = len(history.history['loss'])
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                best_val_acc = max(history.history['val_accuracy'])
                overfitting_gap = final_train_acc - final_val_acc
                
                results_summary.append({
                    "Experimento": exp_name,
                    "Épocas": final_epoch,
                    "Train Acc": f"{final_train_acc:.4f}",
                    "Val Acc": f"{final_val_acc:.4f}",
                    "Best Val Acc": f"{best_val_acc:.4f}",
                    "Overfitting Gap": f"{overfitting_gap:.4f}",
                    "Estado": "✅ COMPLETADO"
                })
                # Evaluar sobrefitting
                if overfitting_gap > 0.1:
                    print(f"   ALERTA: Alto overfitting detectado")
                elif overfitting_gap > 0.05:
                    print(f"   AVISO: Overfitting moderado")
                else:
                    print(f"   Excelente: Overfitting controlado")
                
            except Exception as e:
                print(f"❌ ERROR en experimento {exp_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                results_summary.append({
                    "Experimento": exp_name,
                    "Épocas": "N/A",
                    "Train Acc": "N/A",
                    "Val Acc": "N/A",
                    "Best Val Acc": "N/A",
                    "Overfitting Gap": "N/A",
                    "Estado": f"❌ ERROR: {str(e)[:50]}..."
                })
                continue
    
    # ========== RESUMEN FINAL ==========
    if results_summary:
        # Convertir a DataFrame para mejor visualización
        results_df = pd.DataFrame(results_summary)
        # Guardar resumen
        results_df.to_csv("results/experiments_summary.csv", index=False)
        # Análisis del mejor modelo
        completed_exps = [r for r in results_summary if r["Estado"] == "✅ COMPLETADO"]
        if completed_exps:
            # Encontrar el modelo con mejor val accuracy
            best_exp = max(
                completed_exps, 
                key=lambda x: float(x["Best Val Acc"]) if x["Best Val Acc"] != "N/A" else 0
            )
            
            print(f"\nMEJOR MODELO: {best_exp['Experimento']}")
            print(f"   • Best Val Accuracy: {best_exp['Best Val Acc']}")
            print(f"   • Overfitting Gap: {best_exp['Overfitting Gap']}")
            print(f"   • Ubicación: models/best_model_{best_exp['Experimento']}.h5")
    
    print("\n" + "="*70)
    print("¡TODOS LOS EXPERIMENTOS HAN FINALIZADO!")
    print("="*70)
    print("ARCHIVOS GENERADOS:")
    print("   • Modelos: models/best_model_*.h5")
    print("   • Gráficas: results/*.png")
    print("   • Métricas: results/*.json")
    print("   • Reportes: results/*.txt")
    print("   • Resumen: results/experiments_summary.csv")
    print("="*70)