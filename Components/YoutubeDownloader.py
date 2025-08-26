# Components/YoutubeDownloader.py
from __future__ import annotations
import os
import re
import unicodedata
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Tuple

from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.streams import Stream

from Components.common_ffmpeg import run_ffmpeg_with_progress
from Components.common_utils import create_safe_filename, ensure_directory_exists

logger = logging.getLogger("rich")

def get_video_streams(url: str) -> Tuple[YouTube, List[Stream]]:
    """Initializes YouTube object and returns it along with available video streams."""
    yt = YouTube(url, on_progress_callback=on_progress)
    video_streams = [s for s in yt.streams if s.type == "video"]
    return yt, video_streams

def download_and_merge(
    yt: YouTube,
    vstream: Stream,
    output_dir: Path
) -> str | None:
    """
    Downloads the selected video stream and the best audio, then merges them.
    Returns the path to the final merged video.
    """
    ensure_directory_exists(output_dir)
    base_name = output_dir.name

    # Prefer webm audio if available for better compatibility with copy codec
    audio_streams = [s for s in yt.streams if s.type == "audio"]
    audio_webm = next((a for a in audio_streams if "webm" in (a.mime_type or "")), None)
    astream = audio_webm or (audio_streams[0] if audio_streams else None)
    
    if astream is None:
        logger.error("No audio stream found.")
        return None

    video_path = output_dir / f"video_{base_name}{vstream.subtype and '.' + vstream.subtype or '.mp4'}"
    audio_path = output_dir / f"audio_{base_name}{astream.subtype and '.' + astream.subtype or '.m4a'}"
    out_path = output_dir / f"{base_name}.mp4"

    logger.info(f"Downloading video: '{yt.title}'")
    vstream.download(output_path=str(output_dir), filename=video_path.name)
    logger.info("Downloading audio...")
    astream.download(output_path=str(output_dir), filename=audio_path.name)

    logger.info("Merging video and audio...")
    try:
        # Try to merge with copy codec first (faster)
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=True)
    except Exception as e:
        logger.warning(f"Merge with codec copy failed ({e}), retrying with re-encode (NVENC).")
        _ffmpeg_merge(video_path, audio_path, out_path, try_copy=False)

    logger.info(f"Downloaded and merged video available at: {out_path}")
    return str(out_path)

def _ffmpeg_merge(video_path: Path, audio_path: Path, out_path: Path, try_copy: bool = False):
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
        run_ffmpeg_with_progress(cmd, total_duration=None, label="Merge(copy)")
        return

    # Re-encode video (NVENC) + audio (AAC)
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