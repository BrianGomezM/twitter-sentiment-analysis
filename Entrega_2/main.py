# main.py

from src.preprocess import run_preprocess
from src.run_experiments_lstm import run_all_experiments

def main():
    print("🚀 Iniciando proyecto MEJORADO de clasificación de sentimiento")
    print("✨ Mejoras implementadas:")
    print("   - Regularización avanzada (Dropout, L2, BatchNorm)")
    print("   - Early Stopping y ReduceLROnPlateau") 
    print("   - Balanceo inteligente (no forzar mismo tamaño)")
    print("   - Learning Rate optimizado (0.0008)")
    print("   - Gráficas y métricas mejoradas")
    print("   - Modelo más estable y menos overfitting\n")

    print("📌 Paso 1: Preprocesamiento")
    run_preprocess()

    print("\n📌 Paso 2: Ejecución de entrenamientos LSTM MEJORADOS")
    run_all_experiments()
    #d
    print("\n✅ Entrenamiento mejorado completado.")

if __name__ == "__main__":
    main()