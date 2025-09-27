import emoji
import re
import pandas as pd
def limpiar_texto(texto):
    if texto is None:
        return ""

    # Quitar emojis con la librería emoji
    texto = emoji.replace_emoji(texto, replace=" ")

    # Quitar caracteres no deseados
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.,!?¿¡\s]", " ", texto)

    # Normalizar espacios
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

def mostrar_top_palabras(df, X, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    resultados = []

    # 👇 Tweet número 10 (índice 9 porque empieza en 0)
    i = 9  
    tweet = df.iloc[i]["text_clean"]
    row = X[i].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]

    for idx in top_indices:
        palabra = feature_names[idx]
        peso = row[idx]
        if peso > 0:
            resultados.append({
                "Tweet #": i+1,
                "Texto": tweet,
                "Palabra": palabra,
                "Peso TF-IDF": round(peso, 3)
            })

    tabla = pd.DataFrame(resultados)
    return tabla
