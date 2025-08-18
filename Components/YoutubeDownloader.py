# Components/YoutubeDownloader.py
from __future__ import annotations
import os
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import logging

from pytubefix import YouTube
from pytubefix.cli import on_progress

from Components.common_ffmpeg import run_ffmpeg_with_progress

def _safe_slug(text: str, maxlen: int = 32) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:maxlen] if maxlen > 0 else text

def _date_prefix() -> str:
    return datetime.now().strftime("%y_%m_%d")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _ffmpeg_merge(video_path: Path, audio_path: Path, out_path: Path, try_copy: bool = False):
    """
    Funde video+audio. Si try_copy=True intentará -c copy (ideal si ambos son mp4/h264+aac o webm/vp9+opus).
    Si no, usa NVENC para el video y AAC para audio.
    """
    # Detecta por extensión: muy simple pero suele funcionar
    vext = video_path.suffix.lower()
    aext = audio_path.suffix.lower()

    if try_copy and (
        (vext == ".mp4" and aext in (".m4a", ".mp4")) or
        (vext == ".webm" and aext == ".webm")
    ):
        # Muxeamos con copy (rapidísimo)
        cmd = [
            "ffmpeg", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ]
        run_ffmpeg_with_progress(cmd, total_duration=None, label="Merge(copy)")
        return

    # Re-encode de video (NVENC) + audio AAC
    cmd = [
        "ffmpeg", "-hwaccel", "auto",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "h264_nvenc", "-preset", "p5",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run_ffmpeg_with_progress(cmd, total_duration=None, label="Merge(nvenc)")

def download_youtube_video(url: str) -> str | None:
    """
    Descarga video+audio, guarda en subcarpeta por video y devuelve el MP4 final fusionado.
    Naming: YY_MM_DD_<short_desc> (sin acentos ni caracteres raros).
    """
    yt = YouTube(url, on_progress_callback=on_progress)
    title = yt.title or "video"
    short_desc = _safe_slug(title, 28)
    base = f"{_date_prefix()}_{short_desc}"

    base_dir = Path("videos") / base
    _ensure_dir(base_dir)

    # Mostrar opciones de streams
    streams = yt.streams
    video_streams = [s for s in streams if s.type == "video"]
    audio_streams = [s for s in streams if s.type == "audio"]

    print("Available video streams:")
    idx = 0
    for s in video_streams:
        kind = "Progressive" if s.is_progressive else "Adaptive"
        size_mb = (s.filesize or 0) / (1024 * 1024)
        res = getattr(s, "resolution", None) or f"{getattr(s, 'height', '?')}p"
        print(f"{idx}. Resolution: {res}, Size: {size_mb:.2f} MB, Type: {kind}")
        idx += 1

    choice = int(input("Enter the number of the video stream to download: ").strip())
    vstream = video_streams[choice]

    # Preferimos audio/webm si hay (por copy con webm/vp9); si no, el primero disponible
    audio_webm = next((a for a in audio_streams if "webm" in (a.mime_type or "")), None)
    astream = audio_webm or (audio_streams[0] if audio_streams else None)
    if astream is None:
        logging.error("No audio stream found.")
        return None

    video_path = base_dir / f"video_{base}{vstream.subtype and '.' + vstream.subtype or '.mp4'}"
    audio_path = base_dir / f"audio_{base}{astream.subtype and '.' + astream.subtype or '.m4a'}"
    out_path   = base_dir / f"{base}.mp4"

    print(f"Downloading video: '{yt.title}'")
    vstream.download(output_path=str(base_dir), filename=video_path.name)
    print("Downloading audio...")
    astream.download(output_path=str(base_dir), filename=audio_path.name)

    print("Merging video and audio...")
    try:
        # Intentar copy si contenedor/codec compatibles
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=True)
    except Exception as e:
        logging.warning(f"Merge(copy) failed ({e}), retrying with NVENC.")
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=False)

    print(f"Downloaded: '{yt.title}' to {base_dir} as {out_path.name}")
    logging.info(f"Downloaded video at: {out_path}")
    return str(out_path)
