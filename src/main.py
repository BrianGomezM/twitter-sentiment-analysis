"""
Twitter Sentiment Analysis - Main Script
Universidad del Valle - Redes Neuronales 2025-2

Este script ejecuta el pipeline completo de análisis de sentimientos:
1. Carga y limpieza de datos
2. Preprocesamiento para ML
3. Entrenamiento del modelo MLP
4. Evaluación y visualización de resultados
"""

from data.cleaner import DataCleaner
from data.preprocessor import DataPreprocessor
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import plot_training_history

def main():
    """
    Función principal que ejecuta el pipeline completo
    """
    print("="*60)
    print("🐦 TWITTER SENTIMENT ANALYSIS - MLP")
    print("Universidad del Valle - Redes Neuronales 2025-2")
    print("="*60)
    
    try:
        # 1. LIMPIEZA DE DATOS
        print("\n1️⃣  FASE 1: LIMPIEZA DE DATOS")
        print("-" * 30)
        cleaner = DataCleaner()
        df_clean = cleaner.clean_tweet_data()

        # 2. PREPROCESAMIENTO (sin validación)
        print("\n2️⃣  FASE 2: PREPROCESAMIENTO")
        print("-" * 30)
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, encoder, vectorizer = preprocessor.prepare_data(df_clean)

        # 3. ENTRENAMIENTO DEL MODELO MLP
        print("\n3️⃣  FASE 3: ENTRENAMIENTO DEL MODELO MLP")
        print("-" * 30)
        trainer = ModelTrainer()
        model, history = trainer.train_model(X_train, y_train)  # ya no pasamos X_val ni y_val

        # 4. EVALUACIÓN
        print("\n4️⃣  FASE 4: EVALUACIÓN DEL MODELO")
        print("-" * 30)
        evaluator = ModelEvaluator()
        y_pred = evaluator.evaluate_model(model, X_test, y_test, encoder)
        
        # 4. EVALUACIÓN
        print("\n4️⃣  FASE 4: EVALUACIÓN DEL MODELO")
        print("-" * 30)
        evaluator = ModelEvaluator()
        y_pred = evaluator.evaluate_model(model, X_test, y_test, encoder)
        
        # 5. VISUALIZACIÓN
        print("\n5️⃣  FASE 5: VISUALIZACIÓN DE RESULTADOS")
        print("-" * 30)
        plot_training_history(history)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)

        # Después de entrenar
        final_loss = history.history["loss"][-1]
        final_acc = history.history["accuracy"][-1]

        final_val_loss = history.history["val_loss"][-1]
        final_val_acc = history.history["val_accuracy"][-1]

        

        
    except Exception as e:
        print(f"\n❌ ERROR en el pipeline: {e}")
        raise

if __name__ == "__main__":
    main()