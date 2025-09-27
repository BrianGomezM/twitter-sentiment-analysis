import re
import unicodedata

def limpiar_texto(texto):
    # 1. Eliminar emojis y caracteres especiales (rangos Unicode)
    texto = re.sub(r"["
                   u"\U0001F600-\U0001F64F"  # emoticonos
                   u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
                   u"\U0001F680-\U0001F6FF"  # transporte y mapas
                   u"\U0001F1E0-\U0001F1FF"  # banderas
                   "]+", " ", texto, flags=re.UNICODE)
    
    # 2. Quitar todo lo que no sea letras, números, signos de puntuación básicos
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.,!?¿¡\s]", " ", texto)

    # 3. Normalizar espacios
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

# Ejemplo
ejemplo = "Me encantó 😍 el servicio!!! 💯🔥 #Excelente"
print(limpiar_texto(ejemplo))
