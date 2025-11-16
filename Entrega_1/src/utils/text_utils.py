"""
Utilidades para procesamiento de texto
"""
import emoji
import re

def limpiar_texto(texto):
    """
    Limpia y preprocesa texto de tweets
    
    Args:
        texto (str): Texto original del tweet
        
    Returns:
        str: Texto limpio y normalizado
    """
    if texto is None:
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Quitar emojis
    texto = emoji.replace_emoji(texto, replace=" ")
    
    # Quitar menciones de usuario (@usuario)
    texto = re.sub(r'@\w+', ' ', texto)
    
    # Quitar URLs
    texto = re.sub(r'http\S+', ' ', texto)
    
    # Quitar caracteres especiales, mantener letras, números y signos básicos
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.,!?¿¡\s]", " ", texto)
    
    # Normalizar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    
    return texto