# Components/common_ffmpeg.py
import subprocess, shutil, sys, time, math
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
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def run_ffmpeg_with_progress(cmd: list[str], total_duration: float | None, label: str = "ffmpeg"):
    """
    Ejecuta FFmpeg con -progress pipe:1 y muestra %/ETA/speed en vivo.
    Si no conoces la duración total, pasa None y mostrará solo time/speed.
    """
    ff = _ffmpeg_path()
    # Inyectamos flags de progreso sin romper el cmd original
    full = [ff, "-hide_banner", "-y", "-progress", "pipe:1", "-loglevel", "error"] + cmd[1:] if cmd[0] == ff else \
           [ff, "-hide_banner", "-y", "-progress", "pipe:1", "-loglevel", "error"] + cmd
    start = time.time()

    # Lanzamos en texto (utf-8) para leer líneas clave: out_time_ms, speed
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    last_pct = -1
    last_line = ""
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

            # Parse básico
            if line.startswith("out_time_ms="):
                ms = int(line.split("=",1)[1])
                cur_s = ms / 1_000_000.0
                if total_duration and total_duration > 0:
                    pct = min(100, (cur_s / total_duration) * 100.0)
                else:
                    pct = None
                speed = None
                # leemos speed si aparece enseguida
            elif line.startswith("speed="):
                speed = line.split("=",1)[1]
                # imprimir status
                elapsed = time.time() - start
                if total_duration and total_duration > 0:
                    pct_val = min(100.0, (cur_s / total_duration) * 100.0)
                    if pct_val - (last_pct if last_pct >= 0 else 0) >= 0.5 or pct_val in (0,100):
                        # ETA
                        done = max(0.001, cur_s)
                        rate = done / max(0.001, elapsed)
                        remain = total_duration - done
                        eta = remain / max(1e-6, rate)
                        sys.stdout.write(f"\r[{label}] {pct_val:6.2f}% | {cur_s:7.2f}s/{total_duration:7.2f}s | ETA { _format_eta(eta) } | speed {speed:>6}")
                        sys.stdout.flush()
                        last_pct = pct_val
                else:
                    sys.stdout.write(f"\r[{label}] time {cur_s:7.2f}s | speed {speed:>6}")
                    sys.stdout.flush()
    finally:
        ret = proc.wait()
        # salto de línea para limpiar la línea en curso
        sys.stdout.write("\n")
        sys.stdout.flush()
        if ret != 0:
            raise RuntimeError(f"FFmpeg error ({ret}). Última línea: {last_line}")
