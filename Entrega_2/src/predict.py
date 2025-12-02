import sys
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class ModelPredictor:
    def __init__(self, model_path="models/best_model_EXP2_BALANCEADO.h5", 
                 tokenizer_path="models/tokenizer_EXP2_BALANCEADO.json"):
        # Verificar que existen los archivos
        if not os.path.exists(model_path):
            for file in os.listdir("models"):
                if file.endswith(".h5") and "best_model" in file:
                    print(f"   • {file}")
            sys.exit(1)
        if not os.path.exists(tokenizer_path):
            for file in os.listdir("models"):
                if file.endswith(".json") and "tokenizer" in file:
                    print(f"   • {file}")
            sys.exit(1)
        self.model = load_model(model_path)
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        self.tokenizer = tokenizer_from_json(tokenizer_json)
        self.MAX_LEN = 40
        self.MAX_WORDS = 8000
        self.classes = ["negative", "neutral", "positive"]
        
    def preprocess_text(self, text):
        from preprocess import clean_twitter_optimized
        cleaned_text = clean_twitter_optimized(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, 
                              maxlen=self.MAX_LEN, 
                              padding='post', 
                              truncating='post')
        return cleaned_text, padded
    
    def predict_sentiment(self, text):
        cleaned_text, padded_text = self.preprocess_text(text)
        prediction = self.model.predict(padded_text, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        sentiment = self.classes[class_idx]
        probabilities = {
            self.classes[i]: float(prediction[0][i]) 
            for i in range(len(self.classes))
        }
        
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probabilities,
            "class_index": int(class_idx)
        }
    
    def print_prediction(self, result):
        for sentiment, prob in result['probabilities'].items():
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   • {sentiment:8s}: {bar} {prob*100:6.2f}%")
        
    def analyze_text_features(self, text):
        positive_keywords = ["excellent", "great", "good", "perfect", "best", 
                           "amazing", "wonderful", "fantastic", "awesome",
                           "love", "recommend", "happy", "satisfied", 
                           "friendly", "professional", "helpful", "comfortable"]
        
        negative_keywords = ["bad", "terrible", "worst", "horrible", "awful",
                           "never", "delay", "delayed", "cancelled", "canceled",
                           "lost", "missing", "broken", "failed", "issue",
                           "problem", "error", "rude", "uncomfortable", "disappointed"]
        
        neutral_keywords = ["average", "ok", "okay", "acceptable", "normal",
                          "regular", "standard", "fine", "decent", "adequate",
                          "nothing special", "not bad", "not great", "typical"]
        
        text_lower = text.lower()
        
        pos_words = [w for w in positive_keywords if w in text_lower]
        neg_words = [w for w in negative_keywords if w in text_lower]
        neu_words = [w for w in neutral_keywords if w in text_lower]
        
        if pos_words:
            print(f"      {', '.join(pos_words)}")
        
        if neg_words:
            print(f"      {', '.join(neg_words)}")
        
        if neu_words:
            print(f"      {', '.join(neu_words)}")
        
        # Sugerencia basada en palabras clave
        if len(pos_words) > len(neg_words) and len(pos_words) > len(neu_words):
            print(f"\n   SUGGESTION: Text appears POSITIVE based on keywords")
        elif len(neg_words) > len(pos_words) and len(neg_words) > len(neu_words):
            print(f"\n   SUGGESTION: Text appears NEGATIVE based on keywords")
        elif len(neu_words) > 0 and len(neu_words) >= len(pos_words) and len(neu_words) >= len(neg_words):
            print(f"\n   SUGGESTION: Text appears NEUTRAL based on keywords")
        
        return {
            "positive_words": pos_words,
            "negative_words": neg_words,
            "neutral_words": neu_words
        }
    
    def interactive_mode(self): 
        while True:
            try:
                user_input = input("\nEnter text (or 'exit'/'example'/'analyze'): ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'example':
                    # Ejemplos de prueba EN INGLÉS
                    examples = [
                        # NEGATIVE examples
                        "This is the worst flight of my life, never flying with this airline again!",
                        "My flight was delayed 3 hours, terrible service and no compensation.",
                        "@airline you lost my luggage and your customer service is horrible! URL",
                        "Awful experience. Broken seats and rude crew members.",
                        "Never again! Cancelled flight and no help from staff.",
                        
                        # NEUTRAL examples
                        "The flight was okay, nothing special but we arrived on time.",
                        "Average service, could be better but not terrible either.",
                        "Nothing to complain about, standard airline experience.",
                        "Acceptable flight, seats were fine but entertainment system wasn't working.",
                        "It was an okay flight, not great but not bad either.",
                        
                        # POSITIVE examples
                        "Excellent service from the crew, very professional and friendly!",
                        "Great flight experience, comfortable seats and good entertainment.",
                        "Best airline I've flown with! Will definitely recommend to friends.",
                        "Perfect flight, on time departure and amazing in-flight service.",
                        "Wonderful experience, the staff went above and beyond to help."
                    ]
                    
                    import random
                    try:
                        category = input("\nSelect category (1-3) or press Enter for random: ").strip()
                        if category == "1":
                            example = random.choice(examples[:5])  # Primeros 5 son negativos
                        elif category == "2":
                            example = random.choice(examples[5:10])  # Siguientes 5 son neutrales
                        elif category == "3":
                            example = random.choice(examples[10:])  # Últimos 5 son positivos
                        else:
                            example = random.choice(examples)
                    except:
                        example = random.choice(examples)
                    
                    result = self.predict_sentiment(example)
                    self.print_prediction(result)
                    self.analyze_text_features(example)
                    continue
                
                elif user_input.lower() == 'analyze':
                    # Pedir texto para analizar en detalle
                    analyze_text = input("\n🔍 Enter text to analyze in detail: ").strip()
                    if analyze_text:
                        print(f"\nAnalyzing: '{analyze_text}'")
                        analysis = self.analyze_text_features(analyze_text)
                        
                        # También predecir
                        result = self.predict_sentiment(analyze_text)
                        self.print_prediction(result)
                    continue
                
                elif not user_input:
                    continue
                
                # Hacer predicción
                result = self.predict_sentiment(user_input)
                self.print_prediction(result)
                
                # Preguntar si quiere análisis detallado
                if input("\n🔍 Show detailed text analysis? (y/n): ").lower() == 'y':
                    self.analyze_text_features(user_input)
                
            except KeyboardInterrupt:
                print("\n\nProgram terminated by user")
                break
            except Exception as e:
                print(f" Error: {e}")

def main():
    # Verificar si hay modelos entrenados
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("'models/' directory not found")
        print("   Run first: python main.py --train")
        return
    
    # Buscar modelos disponibles
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith(".h5") and "best_model" in f]
    
    if not model_files:
        print(" No trained models found")
        print("   Run first: python main.py --train")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"   {i}. {model_file}")
    
    # Seleccionar modelo
    if len(model_files) == 1:
        selected_model = model_files[0]
        tokenizer_file = selected_model.replace("best_model_", "tokenizer_").replace(".h5", ".json")
        print(f"\nUsing: {selected_model}")
    else:
        try:
            choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
            selected_model = model_files[choice]
            tokenizer_file = selected_model.replace("best_model_", "tokenizer_").replace(".h5", ".json")
            print(f"\nSelected: {selected_model}")
        except:
            print("\nUsing default model: best_model_EXP2_BALANCEADO.h5")
            selected_model = "best_model_EXP2_BALANCEADO.h5"
            tokenizer_file = "tokenizer_EXP2_BALANCEADO.json"
    
    # Crear predictor
    predictor = ModelPredictor(
        model_path=os.path.join("models", selected_model),
        tokenizer_path=os.path.join("models", tokenizer_file)
    )
    
    # Iniciar modo interactivo
    predictor.interactive_mode()

if __name__ == "__main__":
    main()