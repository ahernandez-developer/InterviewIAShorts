# Components/Edit.py
# Utilidades de edición/IO de video aceleradas con FFmpeg (NVENC si disponible)
# - extract_audio_wav: extrae WAV 16k mono (óptimo para ASR)
# - normalize_video_9x16_base: normaliza el master a altura 1920, yuv420p (9:16-ready)
# - trim_video_ffmpeg: recorta por tiempo (re-encode con NVENC/libx264)
#
# NOTAS:
# - En Windows, usa un build de FFmpeg con NVENC (p.ej. Gyan.dev). Si no hay NVENC, se usa libx264.
# - Para evitar errores de dimensiones (yuv420p exige pares), escala siempre a -2:1920 antes del crop vertical.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> None:
    """Ejecuta FFmpeg/FFprobe con logging sencillo y manejo de errores."""
    print("[ffmpeg] " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        # Propaga el log completo para depurar rápido
        raise RuntimeError(f"FFmpeg error ({proc.returncode}):\n{proc.stdout}")
    # Imprime últimas líneas útiles
    tail = "\n".join(proc.stdout.splitlines()[-10:])
    if tail.strip():
        print(tail)


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError(
            "FFmpeg no encontrado en PATH. Instálalo y asegúrate de poder ejecutarlo desde la consola."
        )
    return ff


def _has_nvenc() -> bool:
    """Detecta si el FFmpeg soporta h264_nvenc (encoder por GPU)."""
    ff = _ffmpeg_path()
    proc = subprocess.run([ff, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return "h264_nvenc" in proc.stdout


def _v_encoder() -> List[str]:
    """Devuelve flags de video para usar NVENC si está disponible, sino libx264."""
    if _has_nvenc():
        # p5 ~ fast (equilibrio calidad/velocidad). Ajusta a p4/p6 si deseas.
        return ["-c:v", "h264_nvenc", "-preset", "p5"]
    # Fallback CPU
    return ["-c:v", "libx264", "-preset", "veryfast"]


def extract_audio_wav(src: str, wav: str, sr: int = 16000) -> None:
    """
    Extrae audio a WAV PCM s16, mono, sr=16k.
    Equivalente a lo que se prefiere para ASR tipo Whisper.
    """
    _ensure_parent(wav)
    ff = _ffmpeg_path()
    cmd = [
        ff, "-y",
        "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        wav,
    ]
    _run(cmd)


def normalize_video_9x16_base(src: str, dst: str, fps: int = 30, v_bitrate: str = "8M", a_bitrate: str = "160k") -> None:
    """
    Normaliza el master a:
      - Altura 1920 (par), ancho ajustado automáticamente a par (scale=-2:1920)
      - fps fijo
      - formato yuv420p (evita problemas con H.264)
      - Audio AAC
    Esta salida sirve como base para el crop vertical posterior 1080x1920.
    """
    _ensure_parent(dst)
    ff = _ffmpeg_path()
    venc = _v_encoder()
    vf = f"scale=-2:1920,fps={fps},format=yuv420p"

    cmd = [
        ff, "-y",
        "-i", src,
        "-vf", vf,
        *venc,
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", str(int(int(v_bitrate.rstrip('M')) * 2)) + "M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", a_bitrate,
        "-movflags", "+faststart",
        dst,
    ]
    _run(cmd)


def trim_video_ffmpeg(
    src: str,
    dst: str,
    start: float,
    end: float,
    fps: int = 30,
    v_bitrate: str = "8M",
    a_bitrate: str = "160k",
) -> None:
    """
    Recorta un segmento [start, end] del video con re-encode (para mantener consistencia).
    - Usa NVENC si está disponible; sino libx264.
    - Mantiene fps fijo y audio AAC.
    """
    assert end > start, "end debe ser > start"
    _ensure_parent(dst)
    ff = _ffmpeg_path()
    venc = _v_encoder()

    duration = max(0.0, end - start)
    vf = f"fps={fps},format=yuv420p"  # homogeniza fps y formato

    cmd = [
        ff, "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", src,
        "-vf", vf,
        *venc,
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", str(int(int(v_bitrate.rstrip('M')) * 2)) + "M",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", a_bitrate,
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        dst,
    ]
    _run(cmd)


# (Opcional) Utilidades adicionales

def copy_audio_track(src_with_audio: str, dst_video_no_audio: str, out_path: str, fps: int = 30,
                     v_bitrate: str = "8M", a_bitrate: str = "160k") -> None:
    """
    Mapea el audio de `src_with_audio` sobre el video de `dst_video_no_audio`.
    Re-encodea el video para asegurar compatibilidad (NVENC/libx264).
    """
    _ensure_parent(out_path)
    ff = _ffmpeg_path()
    venc = _v_encoder()

    cmd = [
        ff, "-y",
        "-i", dst_video_no_audio,   # video
        "-i", src_with_audio,       # audio donor
        "-map", "0:v:0", "-map", "1:a:0",
        *venc,
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", str(int(int(v_bitrate.rstrip('M')) * 2)) + "M",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", a_bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd)


if __name__ == "__main__":
    # Pruebas rápidas manuales (ajusta rutas)
    src_demo = "videos/demo.mp4"
    work = Path("work"); work.mkdir(exist_ok=True)
    norm = work / "normalized.mp4"
    cut = work / "cut.mp4"
    wav = work / "audio.wav"

    if Path(src_demo).exists():
        normalize_video_9x16_base(src_demo, str(norm))
        trim_video_ffmpeg(str(norm), str(cut), start=5.0, end=20.0)
        extract_audio_wav(str(norm), str(wav))
        print("OK")
    else:
        print("Coloca un archivo en videos/demo.mp4 para probar.")
