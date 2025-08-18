# Components/FaceCropYOLO.py
from __future__ import annotations
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import cv2

OUT_W, OUT_H = 1080, 1920
DET_SIZE = 640
CONF_THR = 0.35
EMA_ALPHA = 0.15
FACE_SCALE_BOX = 1.35

def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _open_writer_with_fallback(path: str, fps: float):
    """Intenta mp4v y si falla, prueba avc1 (más compatible en Windows)."""
    fourcc_list = ["mp4v", "avc1"]
    for tag in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(str(path), fourcc, fps, (OUT_W, OUT_H))
        if w.isOpened():
            print(f"[VideoWriter] Opened with fourcc={tag}")
            return w
        else:
            w.release()
    raise RuntimeError("No se pudo abrir VideoWriter para MP4 (mp4v/avc1). Verifica codecs/FFmpeg.")

def _expand_bbox(x, y, w, h, scale, W, H):
    cx, cy = x + w/2.0, y + h/2.0
    nw, nh = w*scale, h*scale
    x1 = int(round(cx - nw/2)); y1 = int(round(cy - nh/2))
    x2 = int(round(cx + nw/2)); y2 = int(round(cy + nh/2))
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(x1+1, min(W, x2));  y2 = max(y1+1, min(H, y2))
    return x1, y1, x2-x1, y2-y1

def _resize_to_h(frame, target_h=OUT_H):
    h, w = frame.shape[:2]
    s = target_h / float(h)
    new_w = int(round(w * s))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR), new_w, target_h, s

# ---------------------------
# Backends de detección
# ---------------------------
class _YoloFaceDetector:
    """YOLOv8-face desde un .pt local; si no hay CUDA seguirá en CPU."""
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.weights = weights_path
        self.model = YOLO(self.weights)
        self.device = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.half = self.device != "cpu"

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = DET_SIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_top = (DET_SIZE - nh) // 2
        pad_left = (DET_SIZE - nw) // 2
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

        res = self.model.predict(
            source=canvas,
            imgsz=DET_SIZE,
            conf=CONF_THR,
            verbose=False,
            device=self.device,
            half=self.half
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return None

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[idx]

        # deshacer padding + escala
        x1 = (x1 - pad_left) / scale
        x2 = (x2 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        y2 = (y2 - pad_top) / scale

        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

class _Res10DnnDetector:
    """Detector facial Res10 SSD (prototxt + caffemodel). CPU, rápido a 300px."""
    def __init__(self, prototxt: str, caffemodel: str, conf_thr: float = 0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf_thr = conf_thr

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        if detections.shape[2] == 0:
            return None

        best = None
        best_score = -1
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.conf_thr:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            if score > best_score:
                best_score = score
                best = (max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2))
        if best is None:
            return None
        x1, y1, x2, y2 = best
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def _pick_detector():
    # 1) YOLO .pt si existe
    weights = os.getenv("FACE_MODEL_PATH", "models/yolov8n-face.pt")
    if Path(weights).exists():
        print(f"[Face] Using YOLO weights: {weights}")
        try:
            return _YoloFaceDetector(weights)
        except Exception as e:
            print(f"[Face] YOLO init failed ({e}), falling back to DNN...")

    # 2) Fallback: DNN Res10 en root
    proto = Path("deploy.prototxt")
    cafe  = Path("res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if not proto.exists() or not cafe.exists():
        raise FileNotFoundError(
            "No se encontró YOLO (.pt) ni los archivos de DNN Res10. "
            "Provee FACE_MODEL_PATH o coloca deploy.prototxt y res10_...caffemodel en el root."
        )
    print(f"[Face] Using DNN Res10: {proto.name}, {cafe.name}")
    return _Res10DnnDetector(str(proto), str(cafe))

# ---------------------------
# Pipeline principal
# ---------------------------
def crop_follow_face_1080x1920_yolo(input_path: str, output_path: str):
    """
    Detección por frame (YOLO o DNN), suavizado EMA y recorte 9:16 centrado en el rostro.
    """
    _ensure_parent(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = _open_writer_with_fallback(output_path, fps)

    detector = _pick_detector()
    smoothed_cx = None
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame is None or frame.ndim != 3:
            # frame corrupto: crea frame negro para mantener timeline
            frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

        # asegurar 3 canales
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frs, W, H, scale = _resize_to_h(frame)

        face = detector.detect(frame)
        if face is None:
            tx, ty, tw, th = W//2 - 200, H//2 - 200, 400, 400
        else:
            x, y, w, h = face
            x = int(round(x * scale)); y = int(round(y * scale))
            w = int(round(w * scale)); h = int(round(h * scale))
            x, y, w, h = _expand_bbox(x, y, w, h, FACE_SCALE_BOX, W, H)
            tx, ty, tw, th = x, y, w, h

        face_cx = tx + tw/2.0
        smoothed_cx = face_cx if smoothed_cx is None else (EMA_ALPHA * face_cx + (1-EMA_ALPHA) * smoothed_cx)

        left = int(round(smoothed_cx - OUT_W/2))
        left = max(0, min(W - OUT_W, left))

        # crop y saneo
        crop = frs[:, left:left+OUT_W]
        if crop.shape[1] != OUT_W:
            canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            if crop.size > 0:
                xoff = (OUT_W - crop.shape[1]) // 2
                canvas[:, xoff:xoff+crop.shape[1]] = crop
            crop = canvas

        # asegurar shape correcto y contiguidad/dtype
        if crop.shape != (OUT_H, OUT_W, 3):
            print(f"[FaceCrop][WARN] Shape inesperado {crop.shape}, corrigiendo…")
            tmp = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            h, w = min(crop.shape[0], OUT_H), min(crop.shape[1], OUT_W)
            tmp[:h, :w] = crop[:h, :w]
            crop = tmp
        crop = np.ascontiguousarray(crop, dtype=np.uint8)

        try:
            writer.write(crop)
        except cv2.error as e:
            # log más visible con info del frame
            print(f"[FaceCrop][ERROR] writer.write() fallo: {e}\n"
                  f"   frame_idx={frame_idx}, crop.shape={crop.shape}, dtype={crop.dtype}, contiguous={crop.flags['C_CONTIGUOUS']}")
            raise

        if frame_idx % 60 == 0:
            elapsed = time.time() - t0
            print(f"[FaceCrop] {frame_idx} frames | {frame_idx/elapsed:.1f} FPS | crop={crop.shape}")

    writer.release()
    cap.release()
    print("[FaceCrop] listo.")

def mux_audio_video_nvenc(video_with_audio: str, video_without_audio: str, dst: str, fps: int = 30, v_bitrate: str = "6M"):
    ff = shutil.which("ffmpeg.exe" if str(Path(sys.executable)).lower().startswith("c:") else "ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg no encontrado")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
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
    import subprocess
    subprocess.run(cmd, check=True)
