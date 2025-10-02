"""
Módulo para limpieza y preprocesamiento de texto
"""
import pandas as pd
from sqlalchemy import text
from database.connection import db
from utils.text_utils import limpiar_texto
from data.loader import DataLoader

class DataCleaner:
    """Clase para limpiar y preparar datos de texto"""
    
    def __init__(self):
        self.engine = db.get_engine()
        self.table_name = "cleaned_tweets"  # ← USAR SOLO MINÚSCULAS
    
    def clean_tweet_data(self):
        """
        Limpia los tweets y los guarda en la base de datos
        """
        # Limpiar tablas duplicadas si existen
        self._cleanup_duplicate_tables()
        
        # Cargar datos originales
        loader = DataLoader()
        df = loader.load_raw_data()
        
        # Aplicar limpieza de texto
        print("Limpiando texto de los tweets...")
        df["text_clean"] = df["text"].apply(limpiar_texto)
        
        # Guardar en base de datos
        self._save_cleaned_data(df)
        
        return df[["tweet_id", "airline_sentiment", "text_clean"]]
    
    def _cleanup_duplicate_tables(self):
        """Elimina tablas duplicadas"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND (table_name = 'cleaned_tweets' OR table_name = 'Cleaned_Tweets')
            """))
            
            existing_tables = [row[0] for row in result]
            
            if existing_tables:
                # Eliminar todas las versiones existentes
                for table in existing_tables:
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
                conn.commit()
                print("Tablas duplicadas eliminadas")
    
    def _save_cleaned_data(self, df):
        """Guarda los datos limpios usando solo minúsculas"""
        # Crear tabla (usar minúsculas consistentemente)
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    tweet_id BIGINT PRIMARY KEY,
                    airline_sentiment VARCHAR(50),
                    text_clean TEXT
                )
            """))
            conn.commit()
        
        # Insertar datos
        df_clean = df[["tweet_id", "airline_sentiment", "text_clean"]]
        df_clean.to_sql(
            self.table_name,
            self.engine,
            if_exists="replace",
            index=False
        )
        
        print(f"Tweets limpiados guardados en la tabla {self.table_name}")