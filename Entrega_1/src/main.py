"""
Twitter Sentiment Analysis - Main Script MEJORADO
"""

from data.cleaner import DataCleaner
from data.preprocessor import DataPreprocessor
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import plot_training_history
from config import get_active_config, ACTIVE_EXPERIMENT


def main():
    """
    Función principal que ejecuta el pipeline completo con configuración dinámica
    """
    config = get_active_config()
    
    print("=" * 60)
    print("🐦 TWITTER SENTIMENT ANALYSIS - MLP DINÁMICO")
    print(f"🎯 Configuración activa: {config['name']} ({ACTIVE_EXPERIMENT})")
    print("Universidad del Valle - Redes Neuronales 2025-2")
    print("=" * 60)
    
    try:
        # 1. LIMPIEZA DE DATOS
        print("\n1️⃣  FASE 1: LIMPIEZA DE DATOS")
        print("-" * 30)
        cleaner = DataCleaner()
        df_clean = cleaner.clean_tweet_data()
        
        # 2. PREPROCESAMIENTO (con config activa)
        print("\n2️⃣  FASE 2: PREPROCESAMIENTO")
        print("-" * 30)
        preprocessor = DataPreprocessor(custom_text_config=config['text'])
        X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer = preprocessor.prepare_data(df_clean)
        
        # 3. ENTRENAMIENTO (con config activa)
        print("\n3️⃣  FASE 3: ENTRENAMIENTO DEL MODELO MLP")
        print("-" * 30)
        trainer = ModelTrainer(custom_config=config)
        model, history = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # 4. EVALUACIÓN
        print("\n4️⃣  FASE 4: EVALUACIÓN DEL MODELO")
        print("-" * 30)
        evaluator = ModelEvaluator()
        y_pred = evaluator.evaluate_model(model, X_test, y_test, encoder)
        
        # 5. VISUALIZACIÓN
        print("\n5️⃣  FASE 5: VISUALIZACIÓN DE RESULTADOS")
        print("-" * 30)
        plot_training_history(history)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución del pipeline: {e}")


if __name__ == "__main__":
    main()
