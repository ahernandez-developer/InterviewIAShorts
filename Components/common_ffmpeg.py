# Components/common_ffmpeg.py
from __future__ import annotations
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

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

def run_ffmpeg_with_progress(
    cmd: list[str], 
    total_duration: float | None, 
    label: str = "ffmpeg",
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None
):
    """
    Ejecuta FFmpeg con -progress pipe:1.
    Si se provee un objeto `rich.progress`, actualiza la tarea correspondiente.
    De lo contrario, imprime el progreso a stdout.
    """
    ff = _ffmpeg_path()

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
        stderr=subprocess.PIPE,
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
            elif line.startswith("time="):
                try:
                    time_str = line.split("=", 1)[1].strip()
                    h, m, s = map(float, time_str.split(":"))
                    cur_s = h * 3600 + m * 60 + s
                except Exception:
                    pass

            elif line.startswith("speed="):
                speed_str = line.split("=", 1)[1]
                
                if progress and task_id is not None and total_duration and total_duration > 0:
                    pct = min(100.0, (cur_s / total_duration) * 100.0)
                    progress.update(task_id, completed=cur_s, description=f"{label} {pct:6.2f}%")
                else:
                    # Fallback a la impresión manual
                    elapsed = time.time() - start
                    if total_duration and total_duration > 0:
                        pct = min(100.0, (cur_s / total_duration) * 100.0)
                        if pct - last_pct_print >= 0.5 or pct in (0.0, 100.0):
                            done = max(0.001, cur_s)
                            rate = done / max(0.001, elapsed)
                            remain = max(0.0, total_duration - done)
                            eta = remain / max(1e-6, rate)
                            sys.stdout.write(
                                f"\r[{label}] {pct:6.2f}% | {cur_s:7.2f}s/{total_duration:7.2f}s | ETA {_format_eta(eta)} | speed {speed_str:>7}"
                            )
                            sys.stdout.flush()
                            last_pct_print = pct
                    else:
                        sys.stdout.write(f"\r[{label}] time {cur_s:7.2f}s | speed {speed_str:>7}")
                        sys.stdout.flush()

    finally:
        ret = proc.wait()
        if not (progress and task_id):
             sys.stdout.write("\n")
             sys.stdout.flush()
        if ret != 0:
            stderr_output = proc.stderr.read()
            raise RuntimeError(f"FFmpeg error ({ret}). Última línea: {last_line}. Stderr: {stderr_output}")
