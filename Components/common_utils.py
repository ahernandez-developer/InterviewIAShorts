# Components/common_utils.py
import unicodedata
import re
from datetime import datetime
from pathlib import Path

def create_safe_filename(text: str, max_len: int = 50, with_date: bool = True) -> str:
    """
    Creates a safe, clean filename from a string.
    Normalizes, removes special characters, truncates, and optionally adds a date prefix.
    """
    # Normalize and remove accents
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alphanumeric with underscore
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    # Remove duplicate underscores
    s = re.sub(r'_+', '_', s)
    # Truncate
    if len(s) > max_len:
        s = s[:max_len].rstrip('_')
    
    if with_date:
        date_prefix = datetime.now().strftime("%y_%m_%d")
        return f"{date_prefix}_{s}"
    else:
        return s

def ensure_directory_exists(path: str | Path) -> None:
    """Checks if a directory exists and creates it if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_parent_directory_exists(file_path: str | Path) -> None:
    """Ensures the parent directory of a file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
