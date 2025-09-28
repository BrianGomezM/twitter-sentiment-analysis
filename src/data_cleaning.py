import pandas as pd
from sqlalchemy import text
from db_connection import get_engine
from utils import limpiar_texto

def clean_data():
    engine = get_engine()

    # 1. Leer tabla original
    df = pd.read_sql("SELECT tweet_id, airline_sentiment, text FROM tweets", engine)

    # 2. Limpiar texto
    df["text_clean"] = df["text"].apply(limpiar_texto)

    # 3. Crear tabla si no existe
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Cleaned_Tweets (
                tweet_id BIGINT PRIMARY KEY,
                airline_sentiment VARCHAR(50),
                text_clean TEXT
            )
        """))
        conn.commit()

    # 4. Insertar datos limpios en bloque
    df_clean = df[["tweet_id", "airline_sentiment", "text_clean"]]
    df_clean.to_sql(
        "Cleaned_Tweets",
        engine,
        if_exists="append",  # inserta sin borrar lo previo
        index=False,
        method="multi"
    )

    print("✅ Tweets limpiados y guardados en la tabla Cleaned_Tweets")
    return df
