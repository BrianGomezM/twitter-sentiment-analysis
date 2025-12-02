# main_rnn.py

from src.preprocess import run_preprocess
from src.run_experiments_rnn import run_all_rnn_experiments

def main():
    print("Iniciando proyecto RNN SIN MEMORIA (TU ENTREGA)")
    print("Mejoras implementadas:")
    print("   - Modelo SimpleRNN (sin memoria interna)")
    print("   - Regularización (Dropout, L2, BatchNorm)")
    print("   - Early Stopping y ReduceLROnPlateau")
    print("   - Balanceo inteligente de dataset")
    print("   - Métricas y gráficas automáticas por experimento\n")

    print("Paso 1: Preprocesamiento")
    run_preprocess()

    print("\nPaso 2: Ejecución de experimentos RNN SIN MEMORIA")
    run_all_rnn_experiments()

    print("\nEntrenamiento RNN completado.\n")

if __name__ == "__main__":
    main()
