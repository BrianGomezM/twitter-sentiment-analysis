"""
Twitter Sentiment Analysis - Model Testing Framework ROUND 3
Hyperparameter Tuning AVANZADO basado en Nadam Optimizer
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import sys
import os

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.cleaner import DataCleaner
from data.preprocessor import DataPreprocessor
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from config import EXPERIMENT_CONFIGS, update_active_config, get_active_config


class AdvancedModelTester:
    """
    Clase para testing AVANZADO - Ronda 3
    """
    
    def __init__(self):
        self.results = []
        self.best_model = None
        self.best_score = 0
        self.previous_best_score = 0.7708  # Score del Nadam Optimizer
        
    def create_advanced_configurations(self):
        """
        Genera configuraciones AVANZADAS basadas en Nadam Optimizer
        """
        # Configuración base del NUEVO MEJOR MODELO (Nadam)
        nadam_base = {
            'text': EXPERIMENT_CONFIGS['wide']['text'].copy(),
            'model': {
                'hidden_units': [1024, 512],
                'dropout_rates': [0.7, 0.6],
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'nadam',  # 🔥 NUEVO OPTIMIZADOR GANADOR
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 5,
                    'restore_best_weights': True
                }
            }
        }
        
        test_configs = {}
        
        # ==================== CONFIGURACIÓN BASELINE ====================
        test_configs['nadam_baseline'] = {
            'name': '🔥 NADAM BASELINE',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        
        # ==================== OPTIMIZACIÓN DE NADAM ====================
        test_configs['nadam_lr_0005'] = {
            'name': '📈 Nadam LR 0.0005',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_lr_0005']['training']['learning_rate'] = 0.0005
        
        test_configs['nadam_lr_0002'] = {
            'name': '📈 Nadam LR 0.0002',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_lr_0002']['training']['learning_rate'] = 0.0002
        
        test_configs['nadam_lr_0008'] = {
            'name': '📈 Nadam LR 0.0008',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_lr_0008']['training']['learning_rate'] = 0.0008
        
        # ==================== ARQUITECTURAS OPTIMIZADAS CON NADAM ====================
        test_configs['nadam_wide_768_384'] = {
            'name': 'Nadam + [768, 384]',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [768, 384],  # Arquitectura más balanceada
                'dropout_rates': [0.5, 0.4],
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': nadam_base['training'].copy()
        }
        
        test_configs['nadam_deep_512_256_128'] = {
            'name': 'Nadam + [512, 256, 128]',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [512, 256, 128],  # Más profunda
                'dropout_rates': [0.4, 0.3, 0.2],
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': nadam_base['training'].copy()
        }
        
        test_configs['nadam_very_wide_1536_768'] = {
            'name': 'Nadam + [1536, 768]',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [1536, 768],  # Más ancha
                'dropout_rates': [0.6, 0.5],
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': nadam_base['training'].copy()
        }
        
        
        test_configs['nadam_low_dropout_04_03'] = {
            'name': '🎯 Nadam Dropout Bajo',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [1024, 512],
                'dropout_rates': [0.4, 0.3],  # Dropout más bajo
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': nadam_base['training'].copy()
        }
        
        test_configs['nadam_high_dropout_08_07'] = {
            'name': '🛡️  Nadam Dropout Alto',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [1024, 512],
                'dropout_rates': [0.8, 0.7],  # Dropout más alto
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': nadam_base['training'].copy()
        }
        
        # ==================== BATCH SIZES OPTIMIZADOS CON NADAM ====================
        test_configs['nadam_batch_16'] = {
            'name': '📦 Nadam Batch 16',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_batch_16']['training']['batch_size'] = 16
        
        test_configs['nadam_batch_24'] = {
            'name': '📦 Nadam Batch 24',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_batch_24']['training']['batch_size'] = 24
        
        test_configs['nadam_batch_48'] = {
            'name': '📦 Nadam Batch 48',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_batch_48']['training']['batch_size'] = 48
        
        test_configs['nadam_batch_64'] = {
            'name': '📦 Nadam Batch 64',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_batch_64']['training']['batch_size'] = 64
        
        # ==================== ESTRATEGIAS DE ENTRENAMIENTO CON NADAM ====================
        test_configs['nadam_epochs_15'] = {
            'name': '⏳ Nadam 15 Épocas',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_epochs_15']['training']['epochs'] = 15
        
        test_configs['nadam_epochs_20'] = {
            'name': '⏳ Nadam 20 Épocas',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_epochs_20']['training']['epochs'] = 20
        
        test_configs['nadam_epochs_8'] = {
            'name': '⏳ Nadam 8 Épocas',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_epochs_8']['training']['epochs'] = 8
        
        # ==================== ACTIVACIONES AVANZADAS CON NADAM ====================
        test_configs['nadam_swish'] = {
            'name': '💫 Nadam + Swish',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_swish']['model']['activation'] = 'swish'
        
        test_configs['nadam_gelu'] = {
            'name': '🧠 Nadam + GELU',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_gelu']['model']['activation'] = 'gelu'
        
        test_configs['nadam_elu'] = {
            'name': '⚡ Nadam + ELU',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_elu']['model']['activation'] = 'elu'
        
        test_configs['nadam_leaky_relu'] = {
            'name': '🌊 Nadam + Leaky ReLU',
            'text': nadam_base['text'].copy(),
            'model': nadam_base['model'].copy(),
            'training': nadam_base['training'].copy()
        }
        test_configs['nadam_leaky_relu']['model']['activation'] = 'leaky_relu'
        
        # ==================== CONFIGURACIONES HÍBRIDAS AVANZADAS ====================
        test_configs['nadam_optimal_hybrid'] = {
            'name': '🏆 Nadam Híbrido Óptimo',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [1024, 512],
                'dropout_rates': [0.4, 0.3],  # Dropout balanceado
                'activation': 'swish',        # Activación moderna
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': {
                'epochs': 12,                 # Épocas balanceadas
                'batch_size': 24,             # Batch size óptimo
                'learning_rate': 0.0005,      # LR más bajo
                'optimizer': 'nadam',
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 5,
                    'restore_best_weights': True
                }
            }
        }
        
        test_configs['nadam_performance'] = {
            'name': '🚀 Nadam Máximo Rendimiento',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [1536, 768],
                'dropout_rates': [0.3, 0.2],  # Dropout muy bajo
                'activation': 'gelu',         # Activación avanzada
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': {
                'epochs': 15,                 # Más épocas
                'batch_size': 16,             # Batch pequeño
                'learning_rate': 0.0002,      # LR muy bajo
                'optimizer': 'nadam',
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 8,            # Más paciencia
                    'restore_best_weights': True
                }
            }
        }
        
        test_configs['nadam_efficient'] = {
            'name': '💡 Nadam Eficiente',
            'text': nadam_base['text'].copy(),
            'model': {
                'hidden_units': [768, 384],
                'dropout_rates': [0.2, 0.1],  # Dropout mínimo
                'activation': 'relu',
                'output_activation': 'softmax',
                'l1': 0.0,
                'l2': 0.0
            },
            'training': {
                'epochs': 8,                  # Menos épocas
                'batch_size': 48,             # Batch más grande
                'learning_rate': 0.0008,      # LR más alto
                'optimizer': 'nadam',
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 4,            # Menos paciencia
                    'restore_best_weights': True
                }
            }
        }
        
        return test_configs
    
    def print_test_header(self, config_name, config):
        """
        Imprime encabezado informativo para cada test
        """
        print("\n" + "="*80)
        print(f"🧪 TESTING AVANZADO: {config['name']}")
        print(f"🔧 Key: {config_name}")
        print("="*80)
        
        print("📋 CONFIGURACIÓN DEL MODELO:")
        print(f"   - Arquitectura: {config['model']['hidden_units']}")
        print(f"   - Activación: {config['model']['activation']}")
        print(f"   - Dropout: {config['model']['dropout_rates']}")
        
        print("⚙️  CONFIGURACIÓN DE ENTRENAMIENTO:")
        print(f"   - Optimizer: {config['training']['optimizer']}")
        print(f"   - Learning Rate: {config['training']['learning_rate']}")
        print(f"   - Batch Size: {config['training']['batch_size']}")
        print(f"   - Epochs: {config['training']['epochs']}")
        print("-"*80)
    
    def run_single_test(self, config_name, config, X_train, y_train, X_val, y_val, X_test, y_test, encoder):
        """
        Ejecuta un solo test con una configuración específica
        """
        start_time = time.time()
        
        try:
            # Actualizar configuración global
            update_active_config('wide')
            global_config = get_active_config()
            
            # Sobrescribir con configuración de test
            for key in ['model', 'training']:
                if key in config:
                    global_config[key].update(config[key])
            
            # Entrenar modelo
            trainer = ModelTrainer(custom_config=global_config)
            model, history = trainer.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluar modelo
            evaluator = ModelEvaluator()
            
            try:
                y_pred = evaluator.evaluate_model(model, X_test, y_test, encoder)
            except TypeError:
                y_pred = evaluator.evaluate_model(model, X_test, y_test, encoder)
            
            # Calcular métricas
            from sklearn.metrics import classification_report, accuracy_score
            
            if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_pred_classes = y_pred.astype(int)
            
            if len(y_test.shape) == 2 and y_test.shape[1] > 1:
                y_true_classes = np.argmax(y_test, axis=1)
            else:
                y_true_classes = y_test.astype(int)
            
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            
            try:
                report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
                f1_macro = report['macro avg']['f1-score']
                f1_weighted = report['weighted avg']['f1-score']
            except Exception as e:
                print(f"⚠️  Error calculando F1 scores: {e}")
                f1_macro = 0.0
                f1_weighted = 0.0
            
            training_time = time.time() - start_time
            
            # Obtener métricas del historial
            final_val_accuracy = None
            if history and hasattr(history, 'history'):
                history_dict = history.history
                for key in ['val_accuracy', 'val_sparse_categorical_accuracy']:
                    if key in history_dict:
                        final_val_accuracy = history_dict[key][-1]
                        break
            
            # Resultados
            result = {
                'config_name': config_name,
                'display_name': config['name'],
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'training_time': training_time,
                'final_val_accuracy': final_val_accuracy,
                'architecture': str(config['model']['hidden_units']),
                'activation': config['model']['activation'],
                'dropout': str(config['model']['dropout_rates']),
                'optimizer': config['training']['optimizer'],
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['training']['batch_size'],
                'epochs': config['training']['epochs'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Imprimir resultados
            print(f"✅ RESULTADOS - {config['name']}:")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   🎯 F1 Macro: {f1_macro:.4f}")
            print(f"   ⚖️  F1 Weighted: {f1_weighted:.4f}")
            print(f"   ⏱️  Tiempo: {training_time:.2f}s")
            if final_val_accuracy:
                print(f"   🔍 Val Accuracy: {final_val_accuracy:.4f}")
            
            # Verificar si es el mejor modelo
            improvement = accuracy - self.previous_best_score
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = result.copy()
                print(f"   🏆 ¡NUEVO MEJOR MODELO! (Accuracy: {accuracy:.4f})")
                if improvement > 0:
                    print(f"   📈 Mejora: +{improvement:.4f} sobre Nadam baseline")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR en {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'config_name': config_name,
                'display_name': config['name'],
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'training_time': 0.0,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def run_advanced_test(self):
        """
        Ejecuta testing avanzado - Ronda 3
        """
        print("🚀 INICIANDO TESTING AVANZADO - RONDA 3")
        print(f"🎯 Objetivo: Superar {self.previous_best_score:.4f} de Nadam")
        print("🎪 Estrategia: Hyperparameter Tuning Avanzado en Nadam")
        print("⏳ Cargando y preparando datos...")
        
        try:
            # Cargar datos
            cleaner = DataCleaner()
            df_clean = cleaner.clean_tweet_data()
            
            # Preprocesar datos
            base_config = EXPERIMENT_CONFIGS['wide']
            preprocessor = DataPreprocessor(custom_text_config=base_config['text'])
            X_train, X_val, X_test, y_train, y_val, y_test, encoder, vectorizer = preprocessor.prepare_data(df_clean)
            
            # Generar configuraciones avanzadas
            test_configs = self.create_advanced_configurations()
            
            print(f"📈 Se evaluarán {len(test_configs)} configuraciones AVANZADAS")
            print("="*80)
            
            # Ejecutar tests
            for i, (config_name, config) in enumerate(test_configs.items(), 1):
                try:
                    self.print_test_header(config_name, config)
                    result = self.run_single_test(
                        config_name, config, X_train, y_train, X_val, y_val, X_test, y_test, encoder
                    )
                    self.results.append(result)
                    
                    print(f"\n📊 Progreso: {i}/{len(test_configs)} completados")
                    print("="*80)
                    
                except Exception as e:
                    print(f"❌ ERROR procesando {config_name}: {str(e)}")
                    continue
            
            # Mostrar resumen final
            self.print_final_summary()
            
            # Exportar resultados
            self.export_results()
            
        except Exception as e:
            print(f"❌ ERROR durante testing avanzado: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def print_final_summary(self):
        """
        Imprime resumen final comparativo
        """
        print("\n" + "="*100)
        print("🎯 RESUMEN FINAL - TESTING AVANZADO RONDA 3")
        print("="*100)
        
        # Ordenar resultados
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        # Top modelos
        print("\n🏆 TOP MEJORES MODELOS AVANZADOS:")
        print("-"*100)
        print(f"{'Pos':<4} {'Configuración':<35} {'Accuracy':<10} {'Mejora':<8} {'Tiempo':<10}")
        print("-"*100)
        
        for i, result in enumerate(sorted_results[:10], 1):
            improvement = result['accuracy'] - self.previous_best_score
            improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            print(f"{i:<4} {result['display_name'][:33]:<35} {result['accuracy']:.4f}    {improvement_str:<8} {result['training_time']:.1f}s")
        
        # Mejor modelo general
        if self.best_model:
            improvement = self.best_score - self.previous_best_score
            print("\n" + "🌟 MEJOR MODELO ENCONTRADO:")
            print("-"*100)
            print(f"🏅 Configuración: {self.best_model['display_name']}")
            print(f"📊 Accuracy: {self.best_model['accuracy']:.4f}")
            print(f"📈 Mejora sobre Nadam: {improvement:.4f}")
            print(f"🎯 F1 Macro: {self.best_model['f1_macro']:.4f}")
            print(f"⚖️  F1 Weighted: {self.best_model['f1_weighted']:.4f}")
            print(f"⏱️  Tiempo: {self.best_model['training_time']:.2f}s")
            
            if improvement > 0:
                print(f"✅ ¡NUEVO RÉCORD! Superado Nadam Optimizer 🎉")
            else:
                print(f"ℹ️  Nadam baseline sigue siendo el mejor")
    
    def export_results(self):
        """
        Exporta resultados avanzados
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        df_results = pd.DataFrame(self.results)
        csv_filename = f"advanced_test_results_{timestamp}.csv"
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        
        json_filename = f"advanced_test_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        if self.best_model:
            best_filename = f"best_advanced_config_{timestamp}.json"
            with open(best_filename, 'w', encoding='utf-8') as f:
                json.dump(self.best_model, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 RESULTADOS AVANZADOS EXPORTADOS:")
        print(f"   📄 CSV: {csv_filename}")
        print(f"   📊 JSON: {json_filename}")
        if self.best_model:
            print(f"   🏆 Mejor configuración: {best_filename}")


def main():
    """
    Función principal para testing avanzado
    """
    print("="*100)
    print("🧪 TWITTER SENTIMENT ANALYSIS - ADVANCED MODEL TESTING")
    print("📈 RONDA 3: Hyperparameter Tuning Avanzado")
    print("🎯 Baseline actual: 77.08% accuracy (Nadam Optimizer)")
    print("🎪 Estrategia: Optimización exhaustiva alrededor de Nadam")
    print("Universidad del Valle - Redes Neuronales 2025-2")
    print("="*100)
    
    tester = AdvancedModelTester()
    tester.run_advanced_test()


if __name__ == "__main__":
    main()