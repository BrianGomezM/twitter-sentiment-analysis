"""
Módulo para manejar conexiones a la base de datos
"""
import psycopg2
from sqlalchemy import create_engine
from config import DB_CONFIG

class DatabaseConnection:
    """Clase para gestionar conexiones a PostgreSQL"""
    
    def __init__(self):
        self.config = DB_CONFIG
    
    def get_connection(self):
        """
        Obtiene conexión directa con psycopg2
        Returns:
            psycopg2.connection: Conexión a la base de datos
        """
        try:
            conn = psycopg2.connect(**self.config)
            return conn
        except Exception as e:
            print(f"❌ Error conectando a la base de datos: {e}")
            raise
    
    def get_engine(self):
        """
        Crea engine de SQLAlchemy para pandas
        Returns:
            sqlalchemy.engine: Engine para operaciones con pandas
        """
        connection_string = (
            f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        engine = create_engine(connection_string)
        return engine

# Instancia global para usar en todo el proyecto
db = DatabaseConnection()