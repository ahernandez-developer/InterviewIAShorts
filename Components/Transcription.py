# Components/Transcription.py
from __future__ import annotations
import os
import sys
import time
import json
import math
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Transcripción (GPU si hay) con faster-whisper
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download  # <- NUEVO

logger = logging.getLogger(__name__)

# Carpeta local de modelos (para cache sin symlinks en Windows)
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Evita warnings de symlinks y problemas de OpenMP en Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def _pick_device_and_compute_type() -> tuple[str, str]:
    """
    Devuelve (device, compute_type) seguros para una RTX 3060.
    - CUDA disponible -> ("cuda", "int8_float16") para buen equilibrio velocidad/calidad.
    - CPU -> ("cpu", "int8")
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "int8_float16"
        return "cpu", "int8"
    except Exception:
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        return device, "int8_float16" if device == "cuda" else "int8"


def _format_time(s: float) -> str:
    if s is None:
        return "NA"
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def _progress_bar(done: float, total: float, width: int = 30) -> str:
    if total <= 0:
        return ""
    ratio = max(0.0, min(1.0, done / total))
    done_w = int(width * ratio)
    return f"[{'█'*done_w}{'.'*(width-done_w)}] {ratio*100:5.1f}%"


def _save_speech_json(segments: List[dict], dst: Path) -> None:
    """
    segments: lista de dicts con: speaker, start, end, text
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Speech diarization JSON saved at: {dst}")


# ------------------------------------------------------------
# (Opcional) Diarización con Pyannote
# ------------------------------------------------------------

def _try_diarization_pyannote(
    wav_path: str,
    hf_token: Optional[str] = None
) -> Optional[List[dict]]:
    """
    Intenta diarizar el audio usando pyannote/speaker-diarization.
    Devuelve una lista de regiones sin texto (solo speaker/start/end).
    Si no se puede (no instalado / falta token / error), retorna None.
    """
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        logger.warning(f"Pyannote no disponible ({e}); sin diarización.")
        return None

    token = hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HUGGINGFACE_TOKEN no configurado. Saltando diarización.")
        return None

    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        diar = pipe(wav_path)

        regions = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            regions.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })

        regions.sort(key=lambda r: r["start"])
        return regions

    except Exception as e:
        logger.warning(f"No se pudo correr diarización pyannote ({e}).")
        return None


def _merge_transcript_with_regions(
    transcript: List[Tuple[str, float, float]],
    regions: Optional[List[dict]]
) -> List[dict]:
    """
    Junta transcripción (text/start/end) con regiones de speaker (start/end)
    Asigna texto al speaker cuyo intervalo solape más.
    Si no hay regions -> asigna todo a SPEAKER_0.
    """
    out = []
    if not regions:
        for (text, start, end) in transcript:
            out.append({
                "speaker": "SPEAKER_0",
                "start": float(start),
                "end": float(end),
                "text": text
            })
        return out

    i = 0
    for (text, s, e) in transcript:
        if s is None or e is None:
            out.append({
                "speaker": "SPEAKER_0",
                "start": float(s or 0.0),
                "end": float(e or (s or 0.0)),
                "text": text
            })
            continue

        best_idx, best_overlap = 0, 0.0
        for j in range(max(0, i - 2), min(len(regions), i + 10)):
            rs, re = regions[j]["start"], regions[j]["end"]
            ov = max(0.0, min(e, re) - max(s, rs))
            if ov > best_overlap:
                best_overlap = ov
                best_idx = j
        i = best_idx

        spk = regions[best_idx]["speaker"]
        out.append({
            "speaker": spk,
            "start": float(s),
            "end": float(e),
            "text": text
        })

    return out


# ------------------------------------------------------------
# Descarga única y carga local del modelo (clave)
# ------------------------------------------------------------

def _ensure_model_local(model_size: str) -> Path:
    """
    Descarga UNA sola vez Systran/faster-whisper-<size> en models/faster-whisper-<size>
    sin symlinks (Windows-friendly). Si ya existe, no descarga nada.
    """
    local_dir = MODELS_DIR / f"faster-whisper-{model_size}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # ¿ya está materializado?
    already = any(f.suffix == ".bin" and f.stat().st_size > 10_000_000
                  for f in local_dir.rglob("*"))
    if already:
        return local_dir

    logger.info(f"Descargando modelo {model_size} a {local_dir} (una sola vez)…")
    snapshot_download(
        repo_id=f"Systran/faster-whisper-{model_size}",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,   # <- importante en Windows
        resume_download=True,
        allow_patterns=["*"],
    )
    return local_dir


# ------------------------------------------------------------
# Transcripción principal
# ------------------------------------------------------------

def transcribeAudio(
    wav_path: str,
    model_size: str = "medium",
    language: Optional[str] = "es",
    beam_size: int = 1,
    vad_filter: bool = True,
    diarization: str = "auto",
    write_speech_json_to: Optional[str] = None,
) -> List[Tuple[str, float, float]]:
    """
    Transcribe un WAV (16k mono recomendado) con faster-whisper.
    Devuelve lista de (text, start, end).
    """
    wav_path = str(wav_path)
    audio_dur = _probe_duration_ffprobe(wav_path)  # solo para progreso “bonito”

    # Dispositivo y cache local del modelo (sin symlinks en Windows)
    device, compute_type = _pick_device_and_compute_type()
    model_dir = MODELS_DIR / f"faster-whisper-{model_size}"
    model_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Transcribing audio...")
    logger.info(
        f"Audio duration: {_format_time(audio_dur)} | device={device}, "
        f"compute_type={compute_type} | lang={language or 'auto'}"
    )

    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(model_dir),
    )

    # Opciones del decoder (todas soportadas en versiones recientes)
    gen_opts = dict(
        language=language,               # None = autodetección
        beam_size=beam_size,             # 1 = greedy (rápido)
        vad_filter=vad_filter,           # filtra silencios
        condition_on_previous_text=True, # contexto entre segmentos
        # best_of y patience aplican solo con sampling; beam_size>1 usa beam search
    )

    # ✅ OJO: transcribe devuelve (segments_generator, info)
    segments_gen, info = model.transcribe(wav_path, **gen_opts)

    segments_out: List[Tuple[str, float, float]] = []
    last_print = time.time()
    t_load = time.time() - t0
    total_done = 0.0

    for seg in segments_gen:
        # Cada `seg` tiene .text, .start, .end
        text = (seg.text or "").strip()
        start = float(seg.start or 0.0)
        end = float(seg.end or start)

        if text:
            segments_out.append((text, start, end))

        # Progreso
        total_done = end
        now = time.time()
        if audio_dur and (now - last_print) > 0.5:
            bar = _progress_bar(total_done, audio_dur, width=28)
            eta_s = max(
                0.0,
                (audio_dur - total_done) * ((now - t0 - t_load) / max(total_done, 1e-3)),
            )
            logger.info(f"[ASR] {bar} | {total_done:6.2f}/{audio_dur:6.2f}s | ETA {_format_time(eta_s)}")
            last_print = now

    # Línea final al 100%
    if audio_dur:
        bar = _progress_bar(audio_dur, audio_dur, width=28)
        logger.info(f"[ASR] {bar} | {audio_dur:6.2f}/{audio_dur:6.2f}s")

    # ---------- (Opcional) diarización + speech.json ----------
    speech_items: Optional[List[dict]] = None
    do_diar = (diarization or "none").lower()
    try:
        if do_diar in ("auto", "pyannote"):
            speech_items = _try_diarization_pyannote(wav_path)
    except Exception as e:
        logger.warning(f"Diarización fallida: {e}")

    merged = _merge_transcript_with_regions(segments_out, speech_items)
    if write_speech_json_to:
        _save_speech_json(merged, Path(write_speech_json_to))

    logger.info(f"Model cached at: {model_dir}")
    logger.info(f"Transcripción completada en {time.time() - t0:.1f}s")
    return segments_out



# ------------------------------------------------------------
# ffprobe para estimar duración (opcional pero útil)
# ------------------------------------------------------------

def _probe_duration_ffprobe(path: str) -> Optional[float]:
    """
    Usa ffprobe (si existe en PATH) para leer la duración del archivo.
    Si falla, devuelve None.
    """
    import shutil
    ffprobe = shutil.which("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe")
    if not ffprobe:
        return None
    try:
        import subprocess, json as _json
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "json", path
        ]
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = _json.loads(p.stdout.decode("utf-8", errors="ignore"))
        streams = data.get("streams", [])
        if not streams:
            return None
        dur = streams[0].get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None
