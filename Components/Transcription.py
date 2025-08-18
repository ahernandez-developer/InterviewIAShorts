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

logger = logging.getLogger(__name__)

# Carpeta local de modelos (para cache sin symlinks en Windows)
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def _pick_device_and_compute_type() -> tuple[str, str]:
    """
    Devuelve (device, compute_type) seguros para una RTX 3060.
    - CUDA disponible -> ("cuda", "int8_float16") para buen equilibrio velocidad/calidad.
    - CPU -> ("cpu", "int8") si hay soporte; si no, "int8" cae a "int8"/'float32' internamente.
    """
    try:
        # faster-whisper se apoya en CTranslate2; la detección "cuda" es estable
        import torch  # solo para informar en logs si el user lo tiene
        if torch.cuda.is_available():
            return "cuda", "int8_float16"
        return "cpu", "int8"
    except Exception:
        # Si torch no está, intentamos de todos modos CUDA; ctranslate2 lo resuelve
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

    # Se requiere token de HF para descargar el pipeline oficial
    token = hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HUGGINGFACE_TOKEN no configurado. Saltando diarización.")
        return None

    try:
        # Modelos actuales suelen llamarse 'pyannote/speaker-diarization-3.1'
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        diar = pipe(wav_path)

        regions = []
        # agrupamos por turnos (cada segmento con speaker y tiempos)
        for turn, _, speaker in diar.itertracks(yield_label=True):
            regions.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })

        # orden por tiempo
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

    # Índice de regiones
    i = 0
    for (text, s, e) in transcript:
        if s is None or e is None:
            # defensa
            out.append({
                "speaker": "SPEAKER_0",
                "start": float(s or 0.0),
                "end": float(e or (s or 0.0)),
                "text": text
            })
            continue

        # buscamos region con mayor solape
        best_idx, best_overlap = 0, 0.0
        for j in range(max(0, i - 2), min(len(regions), i + 10)):
            rs, re = regions[j]["start"], regions[j]["end"]
            ov = max(0.0, min(e, re) - max(s, rs))
            if ov > best_overlap:
                best_overlap = ov
                best_idx = j
        i = best_idx  # aproximación para acelerar

        spk = regions[best_idx]["speaker"]
        out.append({
            "speaker": spk,
            "start": float(s),
            "end": float(e),
            "text": text
        })

    return out


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

    Params:
    - model_size: tiny|base|small|medium|large-v3 etc. (recomendado: 'medium' para es)
    - language: ISO 639-1 ('es', 'en', …) o None para autodetección
    - beam_size: 1 = greedy (rápido); >1 usa beam search
    - vad_filter: True -> ignora silencios/no-speech
    - diarization:
        * "auto" -> intenta pyannote si hay token y paquete instalados, si no "none"
        * "pyannote" -> fuerza pyannote (si falla cae a "none")
        * "none" -> sin diarización
    - write_speech_json_to: ruta para escribir speech.json (opcional)
    """
    wav_path = str(wav_path)
    audio_dur = _probe_duration_ffprobe(wav_path)  # solo para progreso bonito

    device, compute_type = _pick_device_and_compute_type()
    model_dir = MODELS_DIR / f"faster-whisper-{model_size}"
    model_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Transcribing audio...")
    logger.info(
        f"Audio duration: {_format_time(audio_dur)} | "
        f"device={device}, compute_type={compute_type} | lang={language or 'auto'}"
    )

    # Cargar modelo
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(model_dir)
    )
    t_load = time.time() - t0

    # Iterar segmentos con barra de progreso
    segments_out: List[Tuple[str, float, float]] = []
    total_done = 0.0
    last_print = time.time()

    # parámetros razonables
    gen_opts = dict(
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        # para español, el decoder "temperature" default suele ir bien
    )

    for seg in model.transcribe(wav_path, **gen_opts):
        text = seg.text.strip()
        start = float(seg.start or 0.0)
        end = float(seg.end or start)

        if text:
            segments_out.append((text, start, end))

        # progreso
        total_done = end
        now = time.time()
        if audio_dur and (now - last_print) > 0.5:
            bar = _progress_bar(total_done, audio_dur, width=28)
            eta_s = max(0.0, (audio_dur - total_done) * ((now - t0 - t_load) / max(total_done, 1e-3)))
            logger.info(f"[ASR] {bar} | {total_done:6.2f}/{audio_dur:6.2f}s | ETA {_format_time(eta_s)}")
            last_print = now

    # una última línea al 100%
    if audio_dur:
        bar = _progress_bar(audio_dur, audio_dur, width=28)
        logger.info(f"[ASR] {bar} | {audio_dur:6.2f}/{audio_dur:6.2f}s")

    # --------------------------------------------------------
    # (Opcional) diarización + speech.json
    # --------------------------------------------------------
    speech_items: Optional[List[dict]] = None
    do_diar = diarization.lower()
    if do_diar == "auto":
        speech_items = _try_diarization_pyannote(wav_path)
    elif do_diar == "pyannote":
        speech_items = _try_diarization_pyannote(wav_path)
    # "none" -> speech_items = None

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
