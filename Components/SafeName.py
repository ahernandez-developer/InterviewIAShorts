import unicodedata
import re
from datetime import datetime

def safe_name(title: str, max_len: int = 30) -> str:
    # normaliza y elimina acentos
    s = unicodedata.normalize("NFKD", title)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # minúsculas
    s = s.lower()
    # reemplaza cualquier cosa rara por "_"
    s = re.sub(r'[^a-z0-9]+', '_', s)
    # quita underscores duplicados
    s = re.sub(r'_+', '_', s).strip("_")
    # corta a tamaño máximo
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    # agrega fecha
    date = datetime.now().strftime("%y_%m_%d")
    return f"{date}_{s}"
