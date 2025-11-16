# main.py

from src.preprocess import run_preprocess
from src.run_experiments_lstm import run_all_experiments

def main():
    print("🚀 Iniciando proyecto OPTIMIZADO de clasificación de sentimiento")
    print("✨ Mejoras IMPLEMENTADAS:")
    print("   - LSTM Bidireccional para mejor captura de contexto")
    print("   - Regularización avanzada (Dropout 0.4-0.5, L2 0.01)")
    print("   - Early Stopping en val_accuracy con patience=8")
    print("   - Balanceo inteligente por clase específica")
    print("   - ModelCheckpoint para guardar mejores pesos")
    print("   - Learning Rate optimizado con mayor paciencia")
    print("   - Arquitecturas más balanceadas y estables\n")

    print("📌 Paso 1: Preprocesamiento")
    run_preprocess()

    print("\n📌 Paso 2: Ejecución de entrenamientos LSTM OPTIMIZADOS")
    run_all_experiments()

    print("\n✅ Entrenamiento optimizado completado.")

if __name__ == "__main__":
    main()