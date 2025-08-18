# Components/FaceCrop.py
from __future__ import annotations
import os
import math
import cv2
import numpy as np
from pathlib import Path
import subprocess
import shutil
import sys
import logging

# =========================
# Configurables
# =========================
OUT_W, OUT_H = 1080, 1920        # Salida vertical 9:16
DETECT_EVERY_N = 5               # Re-detectar rostro cada N frames (si no hay tracker)
EMA_ALPHA = 0.15                 # Suavizado para la posición X del recorte
FACE_SCALE_BOX = 1.35            # Escala bbox de rostro (margen)
MIN_FACE_SIZE = 64               # Ignorar detecciones muy pequeñas
FACE_DETECTOR = "haar"           # "haar" (por defecto); si pones "none" hará crop centrado

# =========================
# Utilidades
# =========================
def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError("FFmpeg no encontrado en PATH.")
    return ff

def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _create_video_writer(path: str, fps: float):
    # Escribimos H.264 básico desde OpenCV. El MUX final lo hará FFmpeg.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # compatible
    return cv2.VideoWriter(path, fourcc, fps, (OUT_W, OUT_H))

def _resize_to_height(frame, target_h=OUT_H):
    h, w = frame.shape[:2]
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    resized = cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized, new_w, target_h, scale

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))

def _load_haar():
    # Haar cascades vienen con opencv-python (no necesita contrib)
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not Path(haar_path).exists():
        raise RuntimeError("No se encontró haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(haar_path)

def _detect_face_haar(gray, min_size=MIN_FACE_SIZE):
    faces = HAAR.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    # Tomar la cara más grande (asume primer plano)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (int(x), int(y), int(w), int(h))

def _expand_bbox(x, y, w, h, scale, W, H):
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale
    x1 = int(round(cx - nw / 2))
    y1 = int(round(cy - nh / 2))
    x2 = int(round(cx + nw / 2))
    y2 = int(round(cy + nh / 2))
    x1 = _clamp(x1, 0, W - 1)
    y1 = _clamp(y1, 0, H - 1)
    x2 = _clamp(x2, x1 + 1, W)
    y2 = _clamp(y2, y1 + 1, H)
    return x1, y1, x2 - x1, y2 - y1

def _make_tracker():
    # Intentar usar contrib si existe (más estable que redetectar)
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    # Sin contrib: devolvemos None y haremos redetección periódica
    return None

# Cargar Haar una vez (si está configurado)
HAAR = _load_haar() if FACE_DETECTOR == "haar" else None

# =========================
# Pipeline principal
# =========================
def crop_follow_face_1080x1920(input_path: str, output_path: str):
    """
    1) Reescala frame input a altura 1920 manteniendo AR (ancho suele quedar ~3413 si fuente es 1920x1080)
    2) Busca rostro (tracker si hay, o Haar cada N frames)
    3) Suaviza el centro X con EMA para evitar brincos
    4) Recorta ventana vertical 1080x1920 centrada en la cara (clamp a bordes)
    5) Escribe mp4 (H.264 básico). El audio se reinyecta luego con FFmpeg en mux_audio_video_nvenc.
    """
    _ensure_parent(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = _create_video_writer(output_path, fps)

    tracker = None
    has_tracker = False
    track_bbox = None  # (x,y,w,h) en coordenadas del frame reescalado

    smoothed_cx = None  # para suavizado EMA del centro X

    frame_idx = 0
    ok_count = 0
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1) resize to OUT_H
        frame_rs, W, H, scale = _resize_to_height(frame, OUT_H)

        # 2) tracker o redetección
        if tracker is None and (HAAR is not None):
            # redetectar
            gray = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2GRAY)
            det = _detect_face_haar(gray)
            if det and det[2] >= MIN_FACE_SIZE and det[3] >= MIN_FACE_SIZE:
                x, y, w, h = _expand_bbox(*det, FACE_SCALE_BOX, W, H)
                # intentar tracker si existe
                trk = _make_tracker()
                if trk is not None:
                    ok = trk.init(frame_rs, (x, y, w, h))
                    if ok:
                        tracker = trk
                        has_tracker = True
                        track_bbox = (x, y, w, h)
                    else:
                        tracker = None
                        has_tracker = False
                        track_bbox = (x, y, w, h)  # usaremos redetección periódica
                else:
                    # SIN tracker: guardamos bbox y redetectaremos cada N frames
                    track_bbox = (x, y, w, h)
            else:
                # sin detección: centro
                track_bbox = (W // 2 - 200, H // 2 - 200, 400, 400)

        elif tracker is not None and has_tracker:
            ok, box = tracker.update(frame_rs)
            if ok:
                x, y, w, h = [int(round(v)) for v in box]
                x, y, w, h = _expand_bbox(x, y, w, h, FACE_SCALE_BOX, W, H)
                track_bbox = (x, y, w, h)
                ok_count += 1
            else:
                fail_count += 1
                tracker = None  # forzar redetección
                # mantenemos último bbox unos frames
                if track_bbox is None:
                    track_bbox = (W // 2 - 200, H // 2 - 200, 400, 400)

        else:
            # SIN tracker: redetectar cada N frames, si no, mantener bbox previo
            if (frame_idx % DETECT_EVERY_N == 0) and (HAAR is not None):
                gray = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2GRAY)
                det = _detect_face_haar(gray)
                if det and det[2] >= MIN_FACE_SIZE and det[3] >= MIN_FACE_SIZE:
                    x, y, w, h = _expand_bbox(*det, FACE_SCALE_BOX, W, H)
                    track_bbox = (x, y, w, h)
            if track_bbox is None:
                track_bbox = (W // 2 - 200, H // 2 - 200, 400, 400)

        # 3) target center X suavizado
        tx, ty, tw, th = track_bbox
        face_cx = tx + tw / 2.0

        if smoothed_cx is None:
            smoothed_cx = face_cx
        else:
            smoothed_cx = EMA_ALPHA * face_cx + (1 - EMA_ALPHA) * smoothed_cx

        # 4) recorte 1080x1920 a partir del frame reescalado
        # ventana de ancho OUT_W, alto OUT_H, centrada en smoothed_cx
        left = int(round(smoothed_cx - OUT_W / 2))
        left = _clamp(left, 0, W - OUT_W)
        right = left + OUT_W
        crop = frame_rs[:, left:right]

        if crop.shape[1] != OUT_W or crop.shape[0] != OUT_H:
            # llenar con bordes si algo falló
            canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            x_off = (OUT_W - crop.shape[1]) // 2
            canvas[:, x_off:x_off + crop.shape[1]] = crop
            crop = canvas

        writer.write(crop)

    writer.release()
    cap.release()

    logging.info(f"FaceCrop done. Tracker ok={ok_count}, fails={fail_count}, used_tracker={has_tracker}")

def mux_audio_video_nvenc(video_with_audio: str, video_without_audio: str, dst: str, fps: int = 30, v_bitrate: str = "6M"):
    """
    Toma el audio del clip 'video_with_audio' y lo reinyecta en 'video_without_audio' (que viene sin audio).
    Re-encodea el video con NVENC si hace falta alinear formatos.
    """
    _ensure_parent(dst)
    ff = _ffmpeg_path()
    cmd = [
        ff, "-y",
        "-i", video_without_audio,
        "-i", video_with_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "h264_nvenc", "-preset", "p5",
        "-r", str(fps),
        "-b:v", v_bitrate, "-maxrate", v_bitrate, "-bufsize", "12M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        dst
    ]
    # Usamos subprocess simple aquí; ya tendrás %/ETA en el Trim/Merge grande.
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg mux failed: {e}")
