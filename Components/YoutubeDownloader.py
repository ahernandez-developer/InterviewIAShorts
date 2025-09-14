# Components/YoutubeDownloader.py
from __future__ import annotations
import os
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import logging
import sys
from typing import List, Tuple, TYPE_CHECKING

from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.streams import Stream

from Components.common_ffmpeg import run_ffmpeg_with_progress
from Components.common_utils import create_safe_filename, ensure_directory_exists

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = logging.getLogger("rich")

def get_video_streams(url: str) -> Tuple[YouTube, List[Stream]]:
    """Initializes YouTube object and returns it along with available video streams."""
    yt = YouTube(url)
    video_streams = [s for s in yt.streams if s.type == "video"]
    return yt, video_streams

def download_and_merge(
    yt: YouTube,
    vstream: Stream,
    output_dir: Path,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None
) -> str | None:
    """
    Downloads the selected video stream and the best audio, then merges them.
    Returns the path to the final merged video.
    """
    ensure_directory_exists(output_dir)
    base_name = output_dir.name

    audio_streams = [s for s in yt.streams if s.type == "audio"]
    audio_webm = next((a for a in audio_streams if "webm" in (a.mime_type or "")), None)
    astream = audio_webm or (audio_streams[0] if audio_streams else None)
    
    if astream is None:
        logger.error("No audio stream found.")
        return None

    video_path = output_dir / f"video_{base_name}{vstream.subtype and '.' + vstream.subtype or '.mp4'}"
    audio_path = output_dir / f"audio_{base_name}{astream.subtype and '.' + astream.subtype or '.m4a'}"
    out_path = output_dir / f"{base_name}.mp4"

    # --- Download Phase ---
    if progress and task_id:
        progress.update(task_id, description=f"[yellow]Descargando video: {yt.title[:40]}...[/yellow]")
    else:
        logger.info(f"Downloading video: '{yt.title}'")
    vstream.download(output_path=str(output_dir), filename=video_path.name)
    
    if progress and task_id:
        progress.update(task_id, description=f"[yellow]Descargando audio...[/yellow]")
    else:
        logger.info("Downloading audio...")
    astream.download(output_path=str(output_dir), filename=audio_path.name)

    # --- Merge Phase ---
    video_duration = _probe_duration_ffprobe(str(vstream.url))
    if progress and task_id and video_duration:
        progress.update(task_id, total=video_duration, completed=0, description="[cyan]Fusionando archivos...[/cyan]")

    try:
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=True, progress=progress, task_id=task_id, total_duration=video_duration)
    except Exception as e:
        if progress and task_id:
            progress.update(task_id, description=f"[yellow]Fallo en copia directa. Reintentando con re-encode (NVENC)...[/yellow]")
        else:
            logger.warning(f"Merge with codec copy failed ({e}), retrying with re-encode (NVENC).")
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=False, progress=progress, task_id=task_id, total_duration=video_duration)

    if progress and task_id:
        progress.update(task_id, description="[green]Descarga y fusión completadas.[/green]")

    return str(out_path)

def _ffmpeg_merge(
    video_path: Path, 
    audio_path: Path, 
    out_path: Path, 
    try_copy: bool = False,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
    total_duration: float | None = None
):
    """
    Merges video and audio. If try_copy is True, it will attempt to use -c copy.
    Otherwise, it re-encodes using NVENC for video and AAC for audio.
    """
    vext = video_path.suffix.lower()
    aext = audio_path.suffix.lower()

    if try_copy and (
        (vext == ".mp4" and aext in (".m4a", ".mp4")) or
        (vext == ".webm" and aext == ".webm")
    ):
        cmd = [
            "ffmpeg", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ]
        run_ffmpeg_with_progress(cmd, total_duration=total_duration, label="Merge(copy)", progress=progress, task_id=task_id)
        return

    # Re-encode video (NVENC) + audio (AAC)
    cmd = [
        "ffmpeg", "-hwaccel", "auto",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "h264_nvenc", "-preset", "p7",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run_ffmpeg_with_progress(cmd, total_duration=total_duration, label="Merge(nvenc)", progress=progress, task_id=task_id)

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
            "-select_streams", "v:0", # Select video stream for duration
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
