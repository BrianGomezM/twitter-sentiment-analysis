import re
import pandas as pd
import emoji
from sqlalchemy import create_engine, text

# ---------------------------------------
# Función para limpiar texto
# ---------------------------------------
def limpiar_texto(texto):
    if texto is None:
        return ""

    # Quitar emojis
    texto = emoji.replace_emoji(texto, replace=" ")

    # Quitar caracteres no deseados
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.,!?¿¡\s]", " ", texto)

    # Normalizar espacios
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

# ---------------------------------------
# Crear motor SQLAlchemy
# ---------------------------------------
engine = create_engine(
    "postgresql+psycopg2://redes_neuronales_proyecto_user:"
    "qgZaEQHbnkqio5wojYT9VldBH81XYn1k@"
    "dpg-d3bmi5ggjchc738ij1m0-a.oregon-postgres.render.com:5432/"
    "redes_neuronales_proyecto"
)

# ---------------------------------------
# Leer tabla original
# ---------------------------------------
df = pd.read_sql("SELECT tweet_id, airline_sentiment, text FROM tweets", engine)

# ---------------------------------------
# Limpiar texto
# ---------------------------------------
df["text_clean"] = df["text"].apply(limpiar_texto)

# ---------------------------------------
# Crear tabla Cleaned_Tweets si no existe
# ---------------------------------------
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS Cleaned_Tweets (
            tweet_id BIGINT PRIMARY KEY,
            airline_sentiment VARCHAR(50),
            text_clean TEXT
        )
    """))
    conn.commit()

# ---------------------------------------
# Inserción masiva optimizada
# ---------------------------------------
# to_sql con method='multi' hace INSERTs en bloque
df.to_sql(
    "Cleaned_Tweets",
    engine,
    if_exists="append",  # agrega los nuevos registros
    index=False,
    method='multi'       # inserción en bloques
)

print("✅ Tweets limpiados y guardados en la tabla Cleaned_Tweets")
