#!/usr/bin/env python3
"""
Proyecto: Clasificación de Sentimiento en Tweets de Aerolíneas
Entregable 2 - Redes Neuronales con LSTM
Universidad del Valle - Escuela de Ingeniería de Sistemas y Computación
"""
import sys
import os
import pandas as pd

# Añadir src al path para importaciones
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos
from preprocess import run_preprocess
from balance import smart_balance
from run_experiments_lstm import run_all_experiments

def main():
    """Función principal del proyecto"""
    print("=" * 60)
    print("🚀 PROYECTO: CLASIFICACIÓN DE SENTIMIENTO EN TWEETS")
    print("📚 Redes Neuronales - Universidad del Valle")
    print("=" * 60)
    
    print("\n✨ CARACTERÍSTICAS IMPLEMENTADAS:")
    print("   ✅ Preprocesamiento optimizado conservando emociones")
    print("   ✅ Balanceo inteligente de clases")
    print("   ✅ Arquitectura LSTM con regularización")
    print("   ✅ Dropout en LSTM (0.3) para prevenir overfitting")
    print("   ✅ Early Stopping con paciencia de 10 épocas")
    print("   ✅ Split 80/20 entrenamiento/validación")
    print("   ✅ Learning Rate adaptativo (ReduceLROnPlateau)")
    print("   ✅ Gráficas profesionales de métricas")
    print("   ✅ Múltiples experimentos con hiperparámetros\n")

    # Paso 1: Preprocesamiento
    print("📌 PASO 1: PREPROCESAMIENTO DE DATOS")
    print("-" * 40)
    run_preprocess()  # Limpia y guarda datos procesados
    
    # Paso 2: Balanceo
    print("\n📌 PASO 2: BALANCEO INTELIGENTE DE CLASES")
    print("-" * 40)
    df = pd.read_csv("data/processed_tweets.csv")
    df_balanced = smart_balance(df)
    df_balanced.to_csv("data/balanced_tweets.csv", index=False)
    print("✔ Datos balanceados guardados en: data/balanced_tweets.csv")

    # Paso 3: Entrenamiento
    print("\n📌 PASO 3: ENTRENAMIENTO DE MODELOS LSTM")
    print("-" * 40)
    run_all_experiments()

    print("\n" + "=" * 60)
    print("✅ PROYECTO COMPLETADO EXITOSAMENTE")
    print("📊 Modelos guardados como: best_model_*.h5")
    print("📈 Gráficas generadas para análisis")
    print("=" * 60)

if __name__ == "__main__":
    main()