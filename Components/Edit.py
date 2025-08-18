# Components/Edit.py
from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path

from Components.common_ffmpeg import run_ffmpeg_with_progress

def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError("FFmpeg no encontrado en PATH.")
    return ff

def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def extract_audio_wav(src: str, wav: str, sr: int = 16000):
    """
    Extrae WAV mono 16k del contenedor de origen (mp4/m4a/webm/etc).
    """
    ff = _ffmpeg_path()
    _ensure_parent(wav)
    cmd = [
        ff, "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-sample_fmt", "s16",
        wav
    ]
    # No hace falta % aquí (suele ser muy rápido), pero si quieres:
    run_ffmpeg_with_progress(cmd, total_duration=None, label="Audio->WAV")

def trim_video_ffmpeg(src: str, dst: str, start: float, end: float, fps: int = 30,
                      v_bitrate: str = "6M", a_bitrate: str = "160k", copy: bool = False):
    """
    Recorta [start, end] del src.
    - copy=True: usa -c copy (muy rápido) si contenedor/codec lo permite.
    - copy=False: recorta con re-encode (usa NVENC si tu comando global ya lo hace).
    """
    ff = _ffmpeg_path()
    _ensure_parent(dst)
    duration = max(0.0, end - start)
    if copy:
        cmd = [
            ff, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            dst
        ]
        run_ffmpeg_with_progress(cmd, total_duration=duration, label="Trim(copy)")
        return

    # Re-encode (sin filtros pesados aquí; el crop/scale lo hará FaceCrop)
    cmd = [
        ff, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src,
        "-r", str(fps),
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc:v", "vbr",
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", "12M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", a_bitrate,
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        dst
    ]
    run_ffmpeg_with_progress(cmd, total_duration=duration, label="Trim(encode)")
