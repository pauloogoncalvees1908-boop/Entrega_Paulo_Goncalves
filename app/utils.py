import unicodedata
import re

def normalize_text(s: str) -> str:
    """Normaliza unicode (NFKC), colapsa espaços múltiplos, mantém pontuação e emojis.
    Não remove caracteres fora do Basic Multilingual Plane; preserva emojis intactos.
    """
    if s is None:
        return s
    # Normalize unicode (NFKC)
    s = unicodedata.normalize('NFKC', s)
    # Replace newlines/tabs with space
    s = re.sub(r'[\t\r\n]+', ' ', s)
    # Collapse multiple spaces into one
    s = re.sub(r'\s+', ' ', s)
    # Strip leading/trailing spaces
    s = s.strip()
    return s
