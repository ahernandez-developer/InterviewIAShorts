# Components/FaceCropYOLO.py
from __future__ import annotations
import os, json, time, shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

# ================== Salida vertical 9:16 ==================
OUT_W, OUT_H = 1080, 1920

# ================== Detección ==================
DET_SIZE      = 640
CONF_THR      = 0.35
FACE_SCALE    = 1.35   # expandir bbox para no cortar frente/mentón
EDGE_MARGIN   = 20     # margen para no pegar cara a los bordes

# ================== Zoom fijo por segmento ==================
# Queremos que el ancho de la cara ocupe ~RATIO del ancho de la ventana de recorte
FACE_TARGET_RATIO = 0.33
ZOOM_MIN_WIN      = int(OUT_W * 0.85)   # no hacer zoom-in más que esto
ZOOM_MAX_MULT     = 1.70                # zoom-out máximo relativo a OUT_W

# ================== Fallback de segmentación ==================
FALLBACK_SEG_DUR_S = 6.0          # tamaño del bloque cuando no hay diarización
WARMUP_DET_FRAMES  = 12           # nº de frames al inicio del segmento para calcular ROI estable

# ================== Utilidades ==================
def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _resize_to_h(frame, target_h=OUT_H):
    h, w = frame.shape[:2]
    s = target_h / float(h)
    new_w = int(round(w * s))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_LINEAR), new_w, target_h, s

def _expand_bbox(x, y, w, h, scale, W, H):
    cx, cy = x + w/2.0, y + h/2.0
    nw, nh = w*scale, h*scale
    x1 = int(round(cx - nw/2)); y1 = int(round(cy - nh/2))
    x2 = int(round(cx + nw/2)); y2 = int(round(cy + nh/2))
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(x1+1, min(W, x2));  y2 = max(y1+1, min(H, y2))
    return x1, y1, x2-x1, y2-y1

def _open_writer_with_fallback(path: str, fps: float):
    for tag in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(str(path), fourcc, fps, (OUT_W, OUT_H))
        if w.isOpened():
            print(f"[VideoWriter] fourcc={tag} @ {fps:.2f}fps")
            return w
        w.release()
    raise RuntimeError("No se pudo abrir VideoWriter (mp4v/avc1). Verifica codecs/FFmpeg.")

# ================== Detectores ==================
class _YoloFaceDetector:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        # Nota: Ultralytics maneja CUDA internamente; pero podemos pasar 'device'
        self.device = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.half = self.device != "cpu"

    def detect(self, frame_bgr):
        """Devuelve bbox (x,y,w,h) de la mejor cara o None."""
        h, w = frame_bgr.shape[:2]
        scale = DET_SIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_top = (DET_SIZE - nh) // 2
        pad_left = (DET_SIZE - nw) // 2
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized

        res = self.model.predict(
            source=canvas, imgsz=DET_SIZE, conf=CONF_THR,
            verbose=False, device=self.device, half=self.half
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[idx]
        # deshacer padding + escala
        x1 = (x1 - pad_left) / scale; x2 = (x2 - pad_left) / scale
        y1 = (y1 - pad_top) / scale;  y2 = (y2 - pad_top) / scale
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

class _Res10DnnDetector:
    def __init__(self, prototxt: str, caffemodel: str, conf_thr: float = 0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf_thr = conf_thr

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        if detections.shape[2] == 0:
            return None
        best, best_score = None, -1.0
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.conf_thr:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
            if score > best_score:
                best_score, best = score, (x1, y1, x2-x1, y2-y1)
        return best

def _pick_detector():
    weights = os.getenv("FACE_MODEL_PATH", "models/yolov8n-face.pt")
    if Path(weights).exists():
        print(f"[Face] Using YOLO weights: {weights}")
        try:
            return _YoloFaceDetector(weights)
        except Exception as e:
            print(f"[Face] YOLO init failed ({e}), fallback to DNN...")
    proto = Path("deploy.prototxt")
    cafe  = Path("res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if not proto.exists() or not cafe.exists():
        raise FileNotFoundError(
            "No se encontró YOLO (.pt) ni DNN Res10 (deploy.prototxt / res10_*.caffemodel)."
        )
    print(f"[Face] Using DNN Res10: {proto.name}, {cafe.name}")
    return _Res10DnnDetector(str(proto), str(cafe))

# ================== Segmentos (speaker-focus) ==================
@dataclass
class Segment:
    start: float
    end: float
    speaker: str

def _load_segments(speaker_segments_path: Optional[str], duration: float) -> List[Segment]:
    """
    Carga segmentos desde un JSON con [{start, end, speaker}],
    si no existe, crea segmentos por bloques fijos (fallback).
    """
    if speaker_segments_path and Path(speaker_segments_path).exists():
        with open(speaker_segments_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = [Segment(float(d["start"]), float(d["end"]), str(d.get("speaker", "spk"))) for d in data]
        segs = [s for s in segs if s.end > s.start]
        if segs:
            print(f"[Speaker] {len(segs)} segmentos cargados desde {speaker_segments_path}")
            return segs

    # Fallback por bloques fijos
    segs: List[Segment] = []
    t = 0.0
    k = 0
    while t < duration:
        segs.append(Segment(t, min(t + FALLBACK_SEG_DUR_S, duration), f"blk{k}"))
        t += FALLBACK_SEG_DUR_S
        k += 1
    print(f"[Speaker] Fallback por bloques de {FALLBACK_SEG_DUR_S:.1f}s → {len(segs)} segmentos")
    return segs

# ================== Planner de ventana estática ==================
def _window_from_face(face_box, W) -> Tuple[int, int]:
    """
    Calcula ventana de recorte (left, win_w) en el espacio del frame reescalado a OUT_H.
    Usa un zoom fijo en función del ancho de la cara y FACE_TARGET_RATIO.
    """
    min_win = max(ZOOM_MIN_WIN, OUT_W)
    max_win = min(int(OUT_W * ZOOM_MAX_MULT), W)

    if face_box is not None:
        x, y, w, h = face_box
        target_win = int(round(max(w / max(FACE_TARGET_RATIO, 1e-4), min_win)))
    else:
        target_win = int(round(min(OUT_W * 1.15, max_win)))  # contexto si no hay cara

    win_w = int(np.clip(target_win, min_win, max_win))
    cx = W // 2 if face_box is None else int(round(x + w/2))

    # margen para no cortar al borde
    half = win_w // 2
    cx = max(half + EDGE_MARGIN, min(W - half - EDGE_MARGIN, cx))
    left = int(cx - half)
    left = max(0, min(W - win_w, left))
    return left, win_w

def _median_box(boxes: List[Tuple[int,int,int,int]]) -> Optional[Tuple[int,int,int,int]]:
    if not boxes:
        return None
    xs = np.median([b[0] for b in boxes]); ys = np.median([b[1] for b in boxes])
    ws = np.median([b[2] for b in boxes]); hs = np.median([b[3] for b in boxes])
    return (int(xs), int(ys), int(ws), int(hs))

# ================== Pipeline principal (speaker-focus estático) ==================
def crop_follow_face_1080x1920_yolo(
    input_path: str,
    output_path: str,
    speaker_segments_path: Optional[str] = None,
    transition_frames: int = 0  # 0 = corte duro; 6–12 = pequeña transición
):
    """
    1) Lee segmentos (speaker timeline) o crea bloques fijos.
    2) Para cada segmento, estima UNA SOLA vez el ROI (bbox de cara) usando
       N frames de "calentamiento" al inicio del segmento.
    3) Mantiene la cámara estática (misma ventana) durante todo el segmento.
    4) Al cambiar de segmento, hace corte duro o transición corta (opcional).
    """
    _ensure_parent(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if total_frames > 0 else 0.0
    writer = _open_writer_with_fallback(output_path, fps)

    detector = _pick_detector()
    print(f"[Crop] FPS={fps:.2f} | Frames={total_frames} | Dur={duration:.2f}s")

    segments = _load_segments(speaker_segments_path, duration)

    # Helpers de seek/lectura
    def _read_frame_at(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total_frames-1, index)))
        ok, fr = cap.read()
        if not ok or fr is None:
            return None
        if fr.shape[2] == 4:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGRA2BGR)
        frs, W, H, scale = _resize_to_h(fr)   # operar siempre en altura OUT_H
        return fr, frs, W, H, scale

    last_left, last_win_w = None, None  # para transición

    t_global = time.time()
    written = 0

    for i, seg in enumerate(segments):
        s_idx = int(round(seg.start * fps))
        e_idx = int(round(seg.end   * fps))
        if e_idx <= s_idx:
            continue

        # ==== 1) Warmup: calcular bbox estable en los primeros frames del segmento ====
        warm_boxes = []
        probe_end = min(e_idx, s_idx + WARMUP_DET_FRAMES)
        # leer algunos frames del inicio del segmento para estimar ROI
        for fidx in range(s_idx, probe_end):
            sample = _read_frame_at(fidx)
            if sample is None:
                continue
            fr, frs, W, H, scale = sample
            face_raw = detector.detect(fr)  # detección en espacio original
            if face_raw is None:
                continue
            x, y, w, h = face_raw
            # mapear bbox al espacio reescalado (frs)
            x = int(round(x * scale)); y = int(round(y * scale))
            w = int(round(w * scale)); h = int(round(h * scale))
            x, y, w, h = _expand_bbox(x, y, w, h, FACE_SCALE, W, H)
            warm_boxes.append((x, y, w, h))

        face_box = _median_box(warm_boxes)  # puede ser None si nada detectado
        # Calcular ventana fija (left, win_w)
        left, win_w = _window_from_face(face_box, W)

        # ==== 2) Transición con paneo suave opcional entre ventanas ====
        if transition_frames > 0 and last_left is not None and last_win_w is not None:
            for t in range(transition_frames):
                alpha = (t + 1) / float(transition_frames)
                l = int(round((1 - alpha) * last_left  + alpha * left))
                wwin = int(round((1 - alpha) * last_win_w + alpha * win_w))
                base = _read_frame_at(max(s_idx-1, 0))
                if base is None:
                    continue
                _, frs, Wt, Ht, _ = base
                half = wwin // 2
                cx = l + half
                cx = max(half + EDGE_MARGIN, min(Wt - half - EDGE_MARGIN, cx))
                l = max(0, min(Wt - wwin, cx - half))
                crop = frs[:, l:l+wwin]
                crop = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
                writer.write(np.ascontiguousarray(crop))
                written += 1

        # ==== 3) Escribir frames del segmento con cámara ESTÁTICA ====
        for fidx in range(s_idx, e_idx):
            sample = _read_frame_at(fidx)
            if sample is None:
                continue
            _, frs, Wt, Ht, _ = sample
            # clamp de seguridad por si el video cambia ancho en runtime
            l = min(left, max(0, Wt - win_w))
            crop = frs[:, l:l+win_w]
            crop = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            writer.write(np.ascontiguousarray(crop))
            written += 1

        last_left, last_win_w = left, win_w
        print(f"[Seg {i+1}/{len(segments)}] {seg.start:.2f}s→{seg.end:.2f}s | win_w={win_w} | left={left}")

    writer.release()
    cap.release()
    print(f"[FaceCrop] listo. Frames escritos: {written} | t={time.time()-t_global:.2f}s")

# ================== Mux con NVENC (igual que tu helper) ==================
def mux_audio_video_nvenc(video_with_audio: str, video_without_audio: str, dst: str, fps: int = 30, v_bitrate: str = "6M"):
    ff = shutil.which("ffmpeg.exe" if str(Path(os.sys.executable)).lower().startswith("c:") else "ffmpeg")
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
