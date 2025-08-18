# Components/FaceCropYOLO.py
from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import cv2

# ============================
#     PARÁMETROS GLOBALES
# ============================

# Salida vertical 9:16
OUT_W, OUT_H = 1080, 1920

# Detección
DET_SIZE = 640
CONF_THR = 0.35
FACE_SCALE_BOX = 1.50  # expandir bbox para no cortar frente/mentón

# Estabilización de paneo (modo dinámico)
DEADZONE_PX = 24          # zona muerta alrededor del objetivo (px)
MAX_STEP_PX = 18          # avance máximo del centro por frame
SPRING_K    = 0.10        # ganancia del “muelle”
DAMPING     = 0.90        # amortiguación de la velocidad
TARGET_MEDIAN_WIN = 7     # mediana temporal del centro objetivo
CUT_JUMP_THRESH  = 260    # si el objetivo salta mucho, rampa
CUT_RAMP_FRAMES  = 14     # frames de rampa (ease-in-out) al saltar
MISS_HYSTERESIS  = 10     # frames tolerados sin detección
EDGE_MARGIN      = 48     # margen para no pegar al borde

# Zoom dinámico (modo dinámico)
FACE_TARGET_RATIO = 0.28  # cara ≈ 28% del ancho de la ventana
ZOOM_ALPHA        = 0.08  # suavizado EMA del ancho de ventana
ZOOM_MAX_STEP     = 18    # cambio máx. por frame (px)
ZOOM_MIN_WIN      = int(OUT_W * 1.05)  # zoom-in máximo (más conservador)
ZOOM_MAX_WIN_MULT = 1.90  # zoom-out máximo relativo a OUT_W

# Anti-cut (detección de cambio de plano)
CUT_DIFF_THR      = 18.0   # umbral de diferencia media para detectar corte
CUT_COOLDOWN_SEC  = 0.5    # no re-disparar de inmediato


# ============================
#       HELPERS BÁSICOS
# ============================

def _ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _open_writer_with_fallback(path: str, fps: float):
    for tag in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(str(path), fourcc, fps, (OUT_W, OUT_H))
        if w.isOpened():
            print(f"[VideoWriter] fourcc={tag} @ {fps:.2f}fps")
            return w
        w.release()
    raise RuntimeError("No se pudo abrir VideoWriter (mp4v/avc1). Verifica codecs/FFmpeg.")

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


# ============================
#         DETECTORES
# ============================

class _YoloFaceDetector:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.half = self.device != "cpu"
        self.last_conf: float = 0.0
        self.prev_cx: Optional[float] = None  # ayuda a no saltar de sujeto

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
            source=canvas, imgsz=DET_SIZE, conf=CONF_THR,
            verbose=False, device=self.device, half=self.half
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            self.last_conf = 0.0
            return None

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        # Selección: prioriza caja más cercana al último centro; si no hay, usa mayor conf
        if self.prev_cx is not None:
            centers = ((boxes[:,0] + boxes[:,2]) * 0.5)
            idx = int(np.argmin(np.abs(centers - (self.prev_cx * scale + pad_left))))
        else:
            idx = int(np.argmax(confs))

        x1, y1, x2, y2 = boxes[idx]
        self.last_conf = float(confs[idx])

        # deshacer padding + escala
        x1 = (x1 - pad_left) / scale; x2 = (x2 - pad_left) / scale
        y1 = (y1 - pad_top) / scale;  y2 = (y2 - pad_top) / scale
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        cx = (x1 + x2) * 0.5
        self.prev_cx = cx
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

class _Res10DnnDetector:
    def __init__(self, prototxt: str, caffemodel: str, conf_thr: float = 0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf_thr = conf_thr
        self.last_conf: float = 0.0
        self.prev_cx: Optional[float] = None

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        if detections.shape[2] == 0:
            self.last_conf = 0.0
            return None
        best, best_score = None, -1.0
        best_cx = None
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.conf_thr:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
            cx = (x1 + x2) * 0.5
            # prioriza cercanía al último centro si existe
            if self.prev_cx is not None:
                dist = abs(cx - self.prev_cx)
                score += max(0.0, 0.3 - min(dist / (w*0.5), 0.3))  # pequeña bonificación
            if score > best_score:
                best_score, best, best_cx = score, (x1, y1, x2-x1, y2-y1), cx
        self.last_conf = max(best_score, 0.0)
        if best_cx is not None:
            self.prev_cx = best_cx
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
            "No se encontró YOLO (.pt) ni los archivos de DNN Res10. Provee FACE_MODEL_PATH "
            "o coloca deploy.prototxt y res10_...caffemodel en el root."
        )
    print(f"[Face] Using DNN Res10: {proto.name}, {cafe.name}")
    return _Res10DnnDetector(str(proto), str(cafe))


# ============================
#     ESTABILIZADORES (dinámico)
# ============================

class CinematicStabilizer:
    """ Paneo horizontal suave con mediana temporal + spring-damper + deadband + ramp. """
    def __init__(self, width: int):
        self.W = width
        self.cx_smoothed = width / 2.0
        self.vx = 0.0
        self.median_buf = deque(maxlen=TARGET_MEDIAN_WIN)
        self.miss_count = 0
        self.ramp_frames_left = 0
        self.ramp_start = None
        self.ramp_end = None

    def update_target(self, cx_target: float, detected: bool):
        if detected:
            self.miss_count = 0
            self.median_buf.append(cx_target)
            cx_med = np.median(self.median_buf) if self.median_buf else cx_target
        else:
            self.miss_count += 1
            cx_med = np.median(self.median_buf) if self.median_buf else self.cx_smoothed

        delta = abs(cx_med - self.cx_smoothed)
        if delta > CUT_JUMP_THRESH:
            self.ramp_frames_left = CUT_RAMP_FRAMES
            self.ramp_start = self.cx_smoothed
            self.ramp_end = cx_med

        if self.ramp_frames_left > 0:
            t = 1.0 - (self.ramp_frames_left / float(CUT_RAMP_FRAMES))
            t2 = 3*t*t - 2*t*t*t  # ease-in-out cúbica
            cx_goal = (1.0 - t2) * self.ramp_start + t2 * self.ramp_end
            self.ramp_frames_left -= 1
        else:
            cx_goal = cx_med

        err = cx_goal - self.cx_smoothed
        if abs(err) < DEADZONE_PX:
            err = 0.0

        ax = SPRING_K * err
        self.vx = DAMPING * (self.vx + ax)
        self.vx = max(-MAX_STEP_PX, min(MAX_STEP_PX, self.vx))
        self.cx_smoothed += self.vx

        half = OUT_W / 2.0
        min_cx = half + EDGE_MARGIN
        max_cx = self.W - half - EDGE_MARGIN
        self.cx_smoothed = max(min_cx, min(max_cx, self.cx_smoothed))
        return self.cx_smoothed

class ZoomManager:
    """ Mantiene el ancho de la ventana (win_w) según tamaño de cara, movimiento y seguridad en bordes. """
    def __init__(self, frame_width: int):
        self.W = frame_width
        self.win_w = OUT_W

    def update(self, face_box, cx_crop: float, face_conf: float = 1.0, face_vx: float = 0.0):
        W = self.W
        min_win = max(ZOOM_MIN_WIN, OUT_W)
        max_win = min(int(OUT_W * ZOOM_MAX_WIN_MULT), W)

        if face_box is not None:
            x, y, w, h = face_box
            target_by_face = int(round(max(w / max(FACE_TARGET_RATIO, 1e-4), min_win)))
        else:
            target_by_face = int(round(min(OUT_W * 1.10, max_win)))

        # Evita acercarse si hay movimiento o baja confianza
        motion_penalty = 1.0 + min(abs(face_vx) / 24.0, 0.6)   # hasta +60%
        conf_penalty   = 1.0 + max(0.0, 0.6 - min(face_conf, 0.6))  # penaliza conf < 0.6
        target_by_face = int(round(target_by_face * max(motion_penalty, conf_penalty)))

        # Protege cuando la cara está cerca de los bordes de la ventana actual
        protect = self.win_w
        if face_box is not None:
            x, y, w, h = face_box
            face_cx = x + w/2.0
            half = self.win_w / 2.0
            left_edge  = cx_crop - half
            right_edge = cx_crop + half
            left_dist  = face_cx - left_edge
            right_dist = right_edge - face_cx
            edge_thresh = 0.20 * self.win_w
            if left_dist < edge_thresh or right_dist < edge_thresh:
                protect = int(round(min(self.win_w * 1.08, max_win)))

        target = int(round(np.clip(min(target_by_face, protect), min_win, max_win)))

        new_win = int(round(ZOOM_ALPHA * target + (1 - ZOOM_ALPHA) * self.win_w))
        if abs(new_win - self.win_w) > ZOOM_MAX_STEP:
            new_win = self.win_w + np.sign(new_win - self.win_w) * ZOOM_MAX_STEP

        self.win_w = int(round(np.clip(new_win, min_win, max_win)))
        return self.win_w


# ============================
#   CARGA DE TURNOS (speech.json)
# ============================

def _load_turns(speech_json: str | Path, fps: float) -> List[Tuple[float, float, str]]:
    """
    Lee speech.json (lista de items con speaker,start,end,text) y devuelve
    una lista de turnos compactados [(start,end,speaker)] fusionando segmentos
    contiguos del mismo hablante y descartando silencios muy cortos.
    """
    import json
    p = Path(speech_json)
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Ordenar por tiempo y compactar
    items.sort(key=lambda d: float(d.get("start", 0.0)))
    turns: List[Tuple[float, float, str]] = []
    cur_spk = None
    cur_s = None
    cur_e = None

    def _push():
        if cur_spk is None or cur_s is None or cur_e is None:
            return
        dur = cur_e - cur_s
        if dur >= 0.8:  # ignora micro-picos
            turns.append((float(cur_s), float(cur_e), str(cur_spk)))

    for it in items:
        spk = str(it.get("speaker", "SPEAKER_0"))
        s   = float(it.get("start", 0.0))
        e   = float(it.get("end", s))
        if cur_spk is None:
            cur_spk, cur_s, cur_e = spk, s, e
            continue
        if spk == cur_spk and s <= (cur_e + 0.25):  # fusiona si solapa o gap corto
            cur_e = max(cur_e, e)
        else:
            _push()
            cur_spk, cur_s, cur_e = spk, s, e
    _push()
    return turns

def _find_turn_index(turns: List[Tuple[float,float,str]], tsec: float) -> int:
    """Devuelve el índice del turno activo en tsec (o -1 si no hay)."""
    lo, hi = 0, len(turns)-1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, _ = turns[mid]
        if tsec < s: hi = mid - 1
        elif tsec > e: lo = mid + 1
        else: return mid
    return -1


# ============================
#      PIPELINE PRINCIPAL
# ============================

def crop_follow_face_1080x1920_yolo(
    input_path: str,
    output_path: str,
    speech_json: Optional[str] = None,
    static_per_speaker: bool = False,
):
    """
    - Si static_per_speaker=True y speech_json existe: fija una ventana por turno de hablante con transiciones suaves.
    - Si no: usa modo dinámico estabilizado (paneo+zoom) con mejoras.
    """
    _ensure_parent(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = _open_writer_with_fallback(output_path, fps)

    turns = _load_turns(speech_json, fps) if static_per_speaker and speech_json else []
    use_static = static_per_speaker and len(turns) > 0

    detector = _pick_detector()

    frame_idx = 0
    last_face = None

    # --- Estado del modo dinámico (fallback) ---
    stabilizer = None
    zoomer = None

    # --- Estado del modo estático por turno ---
    current_turn = -1
    # Valores finales del ancla para el turno actual
    target_anchor_cx: Optional[float] = None
    target_anchor_win_w: Optional[int] = None
    # Valores de la rampa de transición
    static_ramp_frames_left = 0
    static_ramp_total_frames = 0
    start_anchor_cx, start_anchor_win_w = 0.0, 0
    # Valores a usar en el frame actual (pueden estar en plena rampa)
    current_cx, current_win_w = 0.0, 0

    tracker = None
    tracker_ok = False
    prev_gray = None
    cut_cooldown = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame is None: continue
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frs, W, H, scale = _resize_to_h(frame)

        if current_cx == 0.0: # Inicialización en el primer frame
            current_cx, start_anchor_cx, target_anchor_cx = W / 2.0, W / 2.0, W / 2.0
            current_win_w, start_anchor_win_w, target_anchor_win_w = int(OUT_W * 1.20), int(OUT_W * 1.20), int(OUT_W * 1.20)


        face = detector.detect(frame)
        detected = face is not None

        if detected:
            x, y, w, h = face
            x, y, w, h = (int(round(c * scale)) for c in (x, y, w, h))
            x, y, w, h = _expand_bbox(x, y, w, h, FACE_SCALE_BOX, W, H)
            last_face = (x, y, w, h)
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
                if tracker is not None:
                    tracker_ok = tracker.init(frs, tuple(last_face))
            except AttributeError:
                tracker = None
        elif tracker is not None and tracker_ok:
            tracker_ok, bbox = tracker.update(frs)
            if tracker_ok:
                x, y, w, h = map(int, bbox)
                last_face = _expand_bbox(x, y, w, h, 1.0, W, H)
                detected = True

        tsec = frame_idx / float(fps)
        gray = cv2.cvtColor(frs, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))
            if score > CUT_DIFF_THR and cut_cooldown == 0:
                if stabilizer is not None:
                    stabilizer.ramp_frames_left = CUT_RAMP_FRAMES
                if zoomer is not None:
                    max_win = min(int(OUT_W * ZOOM_MAX_WIN_MULT), W)
                    zoomer.win_w = int(min(zoomer.win_w * 1.15, max_win))
                cut_cooldown = int(fps * CUT_COOLDOWN_SEC)
        prev_gray = gray
        cut_cooldown = max(0, cut_cooldown - 1)

        # ------------------
        # MODO ESTÁTICO
        # ------------------
        if use_static:
            idx = _find_turn_index(turns, tsec)
            if idx != current_turn:
                current_turn = idx
                
                # --- Iniciar transición a un nuevo ancla ---
                start_anchor_cx = current_cx
                start_anchor_win_w = current_win_w
                
                max_win = min(int(OUT_W * ZOOM_MAX_WIN_MULT), W)
                
                if current_turn >= 0:
                    if last_face is not None:
                        fx, fy, fw, fh = last_face
                        target_anchor_cx = fx + fw / 2.0
                        target_w = int(round(max(fw / max(FACE_TARGET_RATIO, 1e-4), int(OUT_W * 1.15))))
                        target_anchor_win_w = int(np.clip(target_w, OUT_W, max_win))
                    else: # Si no hay cara, centrar
                        target_anchor_cx = W / 2.0
                        target_anchor_win_w = int(np.clip(int(OUT_W * 1.20), OUT_W, max_win))
                else: # Si no hay turno, plano general
                    target_anchor_cx = W / 2.0
                    target_anchor_win_w = int(np.clip(int(OUT_W * 1.20), OUT_W, max_win))

                # Duración de la rampa (0.5s)
                static_ramp_total_frames = int(fps * 0.5)
                static_ramp_frames_left = static_ramp_total_frames

            # --- Aplicar rampa de transición si está activa ---
            if static_ramp_frames_left > 0:
                t = 1.0 - (static_ramp_frames_left / float(static_ramp_total_frames))
                t2 = 3*t*t - 2*t*t*t  # Ease-in-out
                current_cx = (1.0 - t2) * start_anchor_cx + t2 * target_anchor_cx
                current_win_w = (1.0 - t2) * start_anchor_win_w + t2 * target_anchor_win_w
                static_ramp_frames_left -= 1
            else:
                current_cx = target_anchor_cx
                current_win_w = target_anchor_win_w

            # Aplicar recorte estático (o en transición)
            half = current_win_w / 2.0
            left = int(round(current_cx - half))
            left = max(0, min(W - int(current_win_w), left))

            if last_face is not None:
                _, fy, _, fh = last_face
                bias = int(min(OUT_H * 0.06, fh * 0.25))
            else:
                bias = int(OUT_H * 0.04)
            frs_bias = cv2.copyMakeBorder(frs, bias, 0, 0, 0, cv2.BORDER_REFLECT_101)

            win = frs_bias[:, left:left+int(current_win_w)]
            crop = cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            writer.write(crop)

        # ------------------
        # MODO DINÁMICO
        # ------------------
        else:
            # ... (el modo dinámico no se modifica)
            if stabilizer is None:
                stabilizer = CinematicStabilizer(W)
            if zoomer is None:
                zoomer = ZoomManager(W)

            if last_face is not None:
                x, y, w, h = last_face
                cx_target = x + w/2.0
                face_for_zoom = last_face
            else:
                if stabilizer.miss_count < MISS_HYSTERESIS:
                    cx_target = stabilizer.cx_smoothed
                    face_for_zoom = None if last_face is None else last_face
                else:
                    cx_target = W / 2.0
                    face_for_zoom = None

            cx = stabilizer.update_target(cx_target, detected)
            lead = np.clip(stabilizer.vx * 1.8, -OUT_W * 0.12, OUT_W * 0.12)
            cx   = np.clip(cx + lead, OUT_W/2 + EDGE_MARGIN, W - OUT_W/2 - EDGE_MARGIN)
            face_conf = getattr(detector, "last_conf", 1.0)
            win_w = zoomer.update(face_for_zoom, cx, face_conf=face_conf, face_vx=stabilizer.vx)
            half  = win_w / 2.0
            left = int(round(cx - half))
            left = max(0, min(W - win_w, left))

            if last_face is not None:
                _, fy, _, fh = last_face
                bias = int(min(OUT_H * 0.06, fh * 0.25))
            else:
                bias = int(OUT_H * 0.04)
            frs_bias = cv2.copyMakeBorder(frs, bias, 0, 0, 0, cv2.BORDER_REFLECT_101)

            win = frs_bias[:, left:left+win_w]
            crop = cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            writer.write(crop)

        if frame_idx % 60 == 0:
            elapsed = round(time.time() - t0, 2)
            mode = "static" if use_static else "dynamic"
            print(f"[FaceCrop:{mode}] f={frame_idx} | {frame_idx/max(elapsed,1e-6):.1f} FPS | W={W}")

    writer.release()
    cap.release()
    print("[FaceCrop] listo.")


# ============================
#      MUX (NVENC)
# ============================

def mux_audio_video_nvenc(video_with_audio: str, video_without_audio: str, dst: str,
                          fps: int = 30, v_bitrate: str = "6M"):
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
