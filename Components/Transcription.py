# Components/Transcription.py
from __future__ import annotations
import os
import time
import logging
from pathlib import Path

from faster_whisper import WhisperModel
import torch

# ========= Config por ENV =========
DEFAULT_MODEL = os.getenv("FW_MODEL_SIZE", "medium").strip()  # small | medium | large-v3
FORCE_LOCAL = os.getenv("FW_FORCE_LOCAL", "0") == "1"         # si 1 => no baja nada de internet
DOWNLOAD_ROOT = Path(os.getenv("FW_DOWNLOAD_ROOT", "models")) # carpeta local en SSD

HEARTBEAT_EVERY = 12  # imprime progreso cada N segmentos
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

def _pick_device_and_compute():
    if torch.cuda.is_available():
        # En RTX 3060: int8_float16 ahorra VRAM y va rápido
        return ("cuda", "int8_float16")
    return ("cpu", "int8")

def transcribeAudio(audio_path: str, model_size: str = DEFAULT_MODEL):
    """
    Transcribe con faster-whisper:
    - Modelos se guardan en carpeta local "models/" (SSD).
    - FW_FORCE_LOCAL=1 => carga solo desde local (si no existe, falla con mensaje).
    - Fallback CPU si cuDNN/CUDA no están listos.
    - Progreso estimado por timestamps de segmentos.
    Devuelve: [[text, start, end], ...]
    """
    try:
        logging.info("Transcribing audio...")
        device, compute_type = _pick_device_and_compute()
        t0 = time.time()

        # Si obligamos local y la carpeta del modelo no existe, avisamos claro.
        model_local_dir = DOWNLOAD_ROOT / f"faster-whisper-{model_size}"
        local_only = FORCE_LOCAL or os.getenv("HF_HUB_OFFLINE", "0") == "1"

        try:
            # Nota: WhisperModel acepta 'download_root' para cache local
            model = WhisperModel(
                model_size if not model_local_dir.exists() else str(model_local_dir),
                device=device,
                compute_type=compute_type,
                download_root=str(DOWNLOAD_ROOT),
                local_files_only=local_only
            )
        except Exception as e:
            logging.warning(f"GPU init or model load failed ({e}). Falling back to CPU INT8 / local-only={local_only}.")
            device, compute_type = ("cpu", "int8")
            model = WhisperModel(
                model_size if not model_local_dir.exists() else str(model_local_dir),
                device=device,
                compute_type=compute_type,
                download_root=str(DOWNLOAD_ROOT),
                local_files_only=local_only
            )

        segments, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        total = float(getattr(info, "duration", 0.0) or 0.0)
        logging.info(
            f"Audio duration: {total:.2f}s | device={device}, compute_type={compute_type} | lang={info.language}"
        )

        out = []
        last_print = 0
        for idx, seg in enumerate(segments, start=1):
            out.append([seg.text.strip(), float(seg.start), float(seg.end)])

            if idx - last_print >= HEARTBEAT_EVERY:
                last_print = idx
                cur = out[-1][2]  # último end
                pct = (cur / total * 100) if total > 0 else 0.0
                elapsed = time.time() - t0
                rate = (cur / max(0.001, elapsed))  # s procesados / s reales
                eta = (total - cur) / max(1e-6, rate) if total > 0 else 0.0
                mm, ss = divmod(int(eta), 60)
                print(f"[ASR] {pct:6.2f}% ({cur:7.2f}/{total:7.2f}s) | ETA {mm:02d}:{ss:02d}")

        if total > 0:
            print(f"[ASR] 100.00% ({total:7.2f}/{total:7.2f}s)")

        # Mensaje útil la primera vez que descarga
        if model_local_dir.exists():
            logging.info(f"Model cached at: {model_local_dir.resolve()}")

        return out

    except Exception as e:
        logging.exception(f"Transcription Error: {e}")
        return []
