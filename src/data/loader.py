"""
Módulo para carga y extracción de datos desde la base de datos
"""
import pandas as pd
from database.connection import db

class DataLoader:
    """Clase para cargar datos desde la base de datos"""
    
    def __init__(self):
        self.engine = db.get_engine()
    
    def load_raw_data(self):
        """
        Carga los datos originales de tweets desde la base de datos
        Returns:
            pandas.DataFrame: DataFrame con tweets originales
        """
        query = "SELECT tweet_id, airline_sentiment, text FROM tweets"
        df = pd.read_sql(query, self.engine)
        print(f"✅ Datos cargados: {len(df)} tweets")
        return df
    
    def load_cleaned_data(self):
        """
        Carga los tweets ya limpiados de la base de datos
        Returns:
            pandas.DataFrame: DataFrame con tweets limpios
        """
        query = "SELECT tweet_id, airline_sentiment, text_clean FROM Cleaned_Tweets"
        df = pd.read_sql(query, self.engine)
        print(f"✅ Datos limpios cargados: {len(df)} tweets")
        return df