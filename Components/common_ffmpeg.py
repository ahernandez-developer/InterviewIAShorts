# Components/common_ffmpeg.py
from __future__ import annotations
import subprocess
import shutil
import sys
import time
from pathlib import Path

def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError("FFmpeg no encontrado en PATH.")
    return ff

def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def run_ffmpeg_with_progress(cmd: list[str], total_duration: float | None, label: str = "ffmpeg"):
    """
    Ejecuta FFmpeg con -progress pipe:1 y muestra %/ETA/speed en vivo.
    Si total_duration es None, muestra time/speed sin %.
    """
    ff = _ffmpeg_path()

    # Si la lista ya empieza por ffmpeg, la reinyectamos con banderas; si no, las anteponemos.
    if Path(cmd[0]).name.lower().startswith("ffmpeg"):
        full = cmd[:1] + ["-hide_banner", "-y", "-progress", "pipe:1", "-loglevel", "error"] + cmd[1:]
    else:
        full = [ff, "-hide_banner", "-y", "-progress", "pipe:1", "-loglevel", "error"] + cmd

    start = time.time()
    cur_s = 0.0
    last_pct_print = -1.0
    last_line = ""

    proc = subprocess.Popen(
        full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Changed to PIPE
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
                continue

            line = line.strip()
            last_line = line

            if line.startswith("out_time_ms="):
                try:
                    ms = int(line.split("=", 1)[1])
                    cur_s = ms / 1_000_000.0
                except Exception:
                    pass

            elif line.startswith("speed="):
                speed = line.split("=", 1)[1]
                elapsed = time.time() - start

                if total_duration and total_duration > 0:
                    pct = min(100.0, (cur_s / total_duration) * 100.0)
                    # Evita spamear: imprime cada ~0.5%
                    if pct - last_pct_print >= 0.5 or pct in (0.0, 100.0):
                        done = max(0.001, cur_s)
                        rate = done / max(0.001, elapsed)  # s procesados / s reales
                        remain = max(0.0, total_duration - done)
                        eta = remain / max(1e-6, rate)
                        sys.stdout.write(
                            f"\r[{label}] {pct:6.2f}% | {cur_s:7.2f}s/{total_duration:7.2f}s | ETA {_format_eta(eta)} | speed {speed:>7}"
                        )
                        sys.stdout.flush()
                        last_pct_print = pct
                else:
                    sys.stdout.write(f"\r[{label}] time {cur_s:7.2f}s | speed {speed:>7}")
                    sys.stdout.flush()

    finally:
        ret = proc.wait()
        sys.stdout.write("\n") # Ensure newline after progress bar
        sys.stdout.flush()
        if ret != 0:
            # Read stderr for error details
            stderr_output = proc.stderr.read()
            raise RuntimeError(f"FFmpeg error ({ret}). Última línea: {last_line}. Stderr: {stderr_output}")
