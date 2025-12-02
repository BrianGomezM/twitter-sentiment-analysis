# src/run_experiments_rnn.py

from src.train_rnn import train_rnn

def run_all_rnn_experiments():

    print("\n Iniciando pruebas automáticas para el MODELO RNN SIN MEMORIA...\n")

    hyperparameter_sets = [
        {"embedding_dim": 50,  "rnn_units": 32,  "dense_units": 16, "epochs": 15,  "batch_size": 64},
        {"embedding_dim": 64,  "rnn_units": 50,  "dense_units": 32, "epochs": 50,  "batch_size": 32},
        {"embedding_dim": 100, "rnn_units": 64,  "dense_units": 64, "epochs": 130, "batch_size": 64}
    ]

    experiment_count = 1

    for params in hyperparameter_sets:
        print(f"\n==============================")
        print(f"EXPERIMENTO RNN #{experiment_count}")
        print(f"Hiperparámetros: {params}")
        print(f"==============================\n")

        try:
            train_rnn(
                embedding_dim=params["embedding_dim"],
                rnn_units=params["rnn_units"],
                dense_units=params["dense_units"],
                epochs=params["epochs"],
                batch_size=params["batch_size"]
            )
        except Exception as e:
            print(f" Error en experimento {experiment_count}: {e}")
            continue

        experiment_count += 1

    print("\n🎉 Todos los experimentos RNN (sin memoria) han finalizado!\n")
