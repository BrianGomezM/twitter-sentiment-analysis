import re
import pandas as pd
from db_connection import get_connection
import emoji
from utils import limpiar_texto

# 1. Conectar
conn = get_connection()
cursor = conn.cursor()

# 2. Leer tabla original
df = pd.read_sql("SELECT tweet_id, airline_sentiment, text FROM tweets", conn)

# 3. Limpiar texto
df["text_clean"] = df["text"].apply(limpiar_texto)

# 4. Crear nueva tabla Cleaned_Tweets (si no existe)
cursor.execute("""
CREATE TABLE IF NOT EXISTS Cleaned_Tweets (
    tweet_id BIGINT PRIMARY KEY,
    airline_sentiment VARCHAR(50),
    text_clean TEXT
)
""")

# 5. Insertar los datos en Cleaned_Tweets
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO Cleaned_Tweets (tweet_id, airline_sentiment, text_clean)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            text_clean = VALUES(text_clean)
    """, (row["tweet_id"], row["airline_sentiment"], row["text_clean"]))

conn.commit()
cursor.close()
conn.close()

print("✅ Tweets limpiados y guardados en la tabla Cleaned_Tweets")
