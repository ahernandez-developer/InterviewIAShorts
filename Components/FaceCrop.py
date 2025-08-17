# Components/FaceCrop.py
# Recorte vertical 1080x1920 con seguimiento horizontal estable y combinación de audio (NVENC si disponible).

from __future__ import annotations

import cv2
import numpy as np
import subprocess
import shutil
import os
from pathlib import Path
from typing import Tuple, Optional


W_OUT, H_OUT = 1080, 1920  # Formato Shorts/Reels/TikTok


# ---------------------------
# Utilidades de ffmpeg/NVENC
# ---------------------------

def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError("FFmpeg no encontrado en PATH.")
    return ff

def _has_nvenc() -> bool:
    ff = _ffmpeg_path()
    proc = subprocess.run([ff, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return "h264_nvenc" in proc.stdout

def _v_encoder_flags() -> list[str]:
    if _has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p5"]
    return ["-c:v", "libx264", "-preset", "veryfast"]


# -----------------------------------------------------
# Detección inicial + tracking y recorte 1080x1920
# -----------------------------------------------------

def _ema(prev: Optional[float], new: float, alpha: float = 0.15) -> float:
    return new if prev is None else (alpha * new + (1.0 - alpha) * prev)

def _initial_face_bbox(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Devuelve (x, y, w, h) de la cara más grande. Si no hay, centra un bbox genérico.
    """
    H, W = frame.shape[:2]
    # Haar cascade estándar (rápido y sin dependencias extra)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        # bbox centrado de emergencia
        w = min(400, W // 3)
        h = min(600, H // 2)
        x = max(0, W // 2 - w // 2)
        y = max(0, H // 2 - h // 2)
        return int(x), int(y), int(w), int(h)
    # cara más grande
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)

def _make_tracker() -> cv2.Tracker:
    """
    Crea un tracker rápido y estable. MOSSE es veloz; CSRT es más robusto si MOSSE no está.
    """
    tracker = None
    try:
        tracker = cv2.legacy.TrackerMOSSE_create()
    except Exception:
        pass
    if tracker is None:
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            pass
    if tracker is None:
        raise RuntimeError("No se pudo crear un tracker (MOSSE/CSRT). Instala opencv-contrib-python.")
    return tracker


def crop_follow_face_1080x1920(input_path: str, output_path: str) -> None:
    """
    Asume que el video de entrada YA fue normalizado a altura 1920 y formato yuv420p
    (ver normalize_video_9x16_base en Components/Edit.py).
    Mantiene H=1920 fija y hace seguimiento SOLO en X para obtener 1080x1920.
    """
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("No se pudo leer el primer frame del video.")

    H, W = frame.shape[:2]

    # Si no está a 1920 de alto, reescala (caso defensivo, lo ideal es normalizar antes)
    if H != H_OUT:
        scale = H_OUT / float(H)
        newW = int(round(W * scale / 2) * 2)  # forzamos par
        frame = cv2.resize(frame, (newW, H_OUT), interpolation=cv2.INTER_AREA)
        W, H = frame.shape[1], frame.shape[0]

    # Detección inicial + tracker
    x, y, w, h = _initial_face_bbox(frame)
    tracker = _make_tracker()
    tracker.init(frame, (x, y, w, h))

    # Posición horizontal suavizada
    cx_smooth: Optional[float] = None

    # writer con dimensiones fijas 1080x1920
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # mp4 compatible
    out = cv2.VideoWriter(output_path, fourcc, fps, (W_OUT, H_OUT))

    # y0 fijo para usar toda la altura (si H==1920 → y0=0)
    if H < H_OUT:
        # letterbox vertical si viniera más bajo (poco probable si normalizaste)
        y0 = 0
    else:
        y0 = (H - H_OUT) // 2  # centra verticalmente si H>1920 (raro)

    # procesar primer frame ya leído
    def _process_and_write(f):
        nonlocal cx_smooth
        ok_t, bb = tracker.update(f)
        if ok_t:
            x_t, y_t, w_t, h_t = map(int, bb)
            cx = x_t + w_t / 2.0
        else:
            cx = W / 2.0  # fallback si se pierde tracking

        cx_smooth = _ema(cx_smooth, cx, alpha=0.15)
        x0 = int(round(cx_smooth - W_OUT / 2.0))
        x0 = max(0, min(x0, W - W_OUT))  # clamp dentro del ancho

        crop = f[y0:y0 + H_OUT, x0:x0 + W_OUT]
        if crop.shape[0] != H_OUT or crop.shape[1] != W_OUT:
            # defensa extra ante bordes
            crop = cv2.copyMakeBorder(
                crop, 0, max(0, H_OUT - crop.shape[0]), 0, max(0, W_OUT - crop.shape[1]),
                cv2.BORDER_REPLICATE
            )
            crop = crop[:H_OUT, :W_OUT]
        out.write(crop)

    _process_and_write(frame)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # mantener resolución constante si hubo rarezas
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        _process_and_write(frame)

    cap.release()
    out.release()


# -----------------------------------------------------
# Combinar audio de un clip con video cropeado (NVENC)
# -----------------------------------------------------

def mux_audio_video_nvenc(video_with_audio: str, video_without_audio: str, dst: str,
                          fps: int = 30, v_bitrate: str = "8M", a_bitrate: str = "160k") -> None:
    """
    Inyecta el audio de `video_with_audio` en `video_without_audio` re-codificando el video con NVENC si está disponible.
    Útil cuando el paso de crop en OpenCV pierde/omite el audio.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_path()
    venc = _v_encoder_flags()

    cmd = [
        ff, "-y",
        "-i", video_without_audio,  # video
        "-i", video_with_audio,     # audio donor
        "-map", "0:v:0", "-map", "1:a:0",
        *venc,
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", str(int(int(v_bitrate.rstrip('M')) * 2)) + "M",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", a_bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst,
    ]
    print("[ffmpeg] " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error ({proc.returncode}):\n{proc.stdout}")
    tail = "\n".join(proc.stdout.splitlines()[-10:])
    if tail.strip():
        print(tail)


if __name__ == "__main__":
    # Prueba manual rápida (ajusta rutas)
    demo_in = "work/cut.mp4"          # clip con audio
    cropped = "work/cropped.mp4"      # salida de crop_follow_face_1080x1920() (sin audio)
    final = "out/Final.mp4"

    if Path(demo_in).exists():
        crop_follow_face_1080x1920(demo_in, cropped)
        mux_audio_video_nvenc(video_with_audio=demo_in, video_without_audio=cropped, dst=final)
        print("OK:", final)
    else:
        print("Coloca un clip en work/cut.mp4 para probar.")
