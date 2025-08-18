# tools/prefetch_models.py
"""
Prefetch de modelos faster-whisper a carpeta local "models/".
Uso:
  set FW_MODEL_SIZE=medium   # o small, large-v3
  python tools/prefetch_models.py
Opcional:
  set HF_TOKEN=xxxxx         # si tu cuenta de HF necesita token
Requiere:
  pip install huggingface_hub
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

SIZE = os.getenv("FW_MODEL_SIZE", "medium").strip()  # small|medium|large-v3
DOWNLOAD_ROOT = Path(os.getenv("FW_DOWNLOAD_ROOT", "models"))
REPO_ID = f"Systran/faster-whisper-{SIZE}"
LOCAL_DIR = DOWNLOAD_ROOT / f"faster-whisper-{SIZE}"
HF_TOKEN = os.getenv("HF_TOKEN", None)

def main():
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"⏬ Prefetching {REPO_ID} → {LOCAL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,  # en Windows evita warnings por symlinks
        token=HF_TOKEN
    )
    print(f"✅ Modelo listo en: {LOCAL_DIR.resolve()}")

if __name__ == "__main__":
    main()
