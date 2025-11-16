# src/run_experiments_lstm.py

from src.train_lstm import train_lstm

def run_all_experiments():
    print("\n🚀 Iniciando pruebas automáticas LSTM MEJORADAS...\n")

    # ============================================================
    # 1. Métodos de limpieza a evaluar (MISMO)
    # ============================================================
    cleaning_methods = [
        "clean_minimal",
        "clean_standard", 
        "clean_aggressive"
    ]

    # ============================================================
    # 2. Hiperparámetros OPTIMIZADOS para mejor estabilidad
    # ============================================================
    hyperparameter_sets = [
        # Configuración CONSERVADORA - mejor generalización
        {"embedding_dim": 64,  "lstm_units": 48,  "dense_units": 24, "epochs": 25, "batch_size": 64},
        
        # Configuración BALANCEADA - punto óptimo
        {"embedding_dim": 80,  "lstm_units": 64,  "dense_units": 32, "epochs": 30, "batch_size": 48},
        
        # Configuración para TEXTOS COMPLEJOS
        {"embedding_dim": 100, "lstm_units": 80, "dense_units": 40, "epochs": 35, "batch_size": 32}
    ]

    # ============================================================
    # 3. Ejecutar todos los experimentos MEJORADOS
    # ============================================================
    experiment_count = 1

    for cleaning in cleaning_methods:
        for params in hyperparameter_sets:
            print(f"\n==============================")
            print(f"🔬 EXPERIMENTO MEJORADO #{experiment_count}")
            print(f"Limpieza: {cleaning}")
            print(f"Hiperparámetros: {params}")
            print(f"==============================\n")

            try:
                train_lstm(
                    cleaning_method=cleaning,
                    embedding_dim=params["embedding_dim"],
                    lstm_units=params["lstm_units"],
                    dense_units=params["dense_units"],
                    epochs=params["epochs"],
                    batch_size=params["batch_size"]
                )
            except Exception as e:
                print(f"❌ Error en experimento {experiment_count}: {e}")
                continue

            experiment_count += 1

    print("\n🎉 Todos los experimentos MEJORADOS han finalizado!\n")