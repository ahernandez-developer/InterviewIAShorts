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
from OneEuroFilter import OneEuroFilter

# ============================
#     PARÁMETROS GLOBALES
# ============================

# Salida vertical 9:16
OUT_W, OUT_H = 1080, 1920
TARGET_ASPECT_RATIO = OUT_W / OUT_H # 0.5625 for 9:16

# Detección
DET_SIZE = 640
CONF_THR = 0.35
FACE_SCALE_BOX = 1.50  # expandir bbox para no cortar frente/mentón

# --- Parámetros del One Euro Filter ---
# Un valor más bajo en min_cutoff aumenta el suavizado (y el lag).
# Un valor más alto en beta permite que el filtro reaccione más rápido a cambios bruscos.
POS_MIN_CUTOFF = 0.5
POS_BETA = 0.5
ZOOM_MIN_CUTOFF = 0.8
ZOOM_BETA = 1.0

# --- Parámetros de Composición de Escena ---
FACE_TARGET_RATIO = 0.28  # La cara debe ocupar ~28% del ancho del cuadro
EDGE_MARGIN = 48
ZOOM_MIN_WIN = int(OUT_W * 1.05)
ZOOM_MAX_WIN_MULT = 1.90

# Anti-cut (detección de cambio de plano)
CUT_DIFF_THR = 18.0
CUT_COOLDOWN_SEC = 0.5


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
#     ESTABILIZADOR (One Euro Filter)
# ============================

class OneEuroStabilizer:
    """Aplica OneEuroFilter a la posición y tamaño de la cara para un seguimiento suave."""
    def __init__(self, freq: float, frame_width: int, frame_height: int):
        self.W = frame_width
        self.H = frame_height
        
        # Filtros para la posición del centro de la cara
        self.filter_x = OneEuroFilter(freq, POS_MIN_CUTOFF, POS_BETA, 1.0)
        self.filter_y = OneEuroFilter(freq, POS_MIN_CUTOFF, POS_BETA, 1.0)
        
        # Filtro para el ancho de la ventana de recorte (controla el zoom)
        self.filter_win_w = OneEuroFilter(freq, ZOOM_MIN_CUTOFF, ZOOM_BETA, 1.0)

        # Estado interno
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0
        self.win_w = OUT_W * 1.2

    def update(self, face_box: Optional[Tuple[int, int, int, int]], t: float) -> Tuple[float, float, float]:
        """
        Actualiza el estabilizador con la nueva detección de cara.
        Devuelve (centro_x_suavizado, centro_y_suavizado, ancho_ventana_suavizado).
        """
        if face_box is not None:
            x, y, w, h = face_box
            # El objetivo es el centro de la cara detectada
            target_cx = x + w / 2.0
            target_cy = y + h / 2.0
            # El objetivo del zoom es mantener la cara a un tamaño constante en pantalla
            target_win_w = w / max(FACE_TARGET_RATIO, 1e-4)
        else:
            # Si no hay cara, el objetivo es el estado actual (mantener la cámara quieta)
            target_cx = self.cx
            target_cy = self.cy
            target_win_w = self.win_w

        # Aplicar filtros de One Euro
        self.cx = self.filter_x(target_cx, t)
        self.cy = self.filter_y(target_cy, t)
        self.win_w = self.filter_win_w(target_win_w, t)

        # --- Restricciones y Lógica de Composición ---
        # Asegurar que el zoom no sea ni muy extremo ni muy pequeño
        max_win = min(int(OUT_W * ZOOM_MAX_WIN_MULT), self.W)
        min_win = max(ZOOM_MIN_WIN, OUT_W)
        self.win_w = np.clip(self.win_w, min_win, max_win)

        # Asegurar que el cuadro de recorte no se salga de los bordes del frame
        half_w = self.win_w / 2.0
        current_win_h = self.win_w / TARGET_ASPECT_RATIO
        half_h = current_win_h / 2.0
        
        min_cx = half_w + EDGE_MARGIN
        max_cx = self.W - half_w - EDGE_MARGIN
        self.cx = np.clip(self.cx, min_cx, max_cx)

        min_cy = half_h + EDGE_MARGIN
        max_cy = self.H - half_h - EDGE_MARGIN
        self.cy = np.clip(self.cy, min_cy, max_cy)

        return self.cx, self.cy, self.win_w


# ============================
#   CARGA DE TURNOS (speech.json)
# ============================

def _load_turns(speech_json: str | Path, fps: float, highlight_start_sec: float) -> List[Tuple[float, float, str]]:
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
        s   = float(it.get("start", 0.0)) - highlight_start_sec # Adjust start time
        e   = float(it.get("end", s)) - highlight_start_sec   # Adjust end time
        # Ensure times are not negative after adjustment
        s = max(0.0, s)
        e = max(0.0, e)
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
    tolerance = 0.15 # seconds, to account for small gaps or rounding issues

    best_match_idx = -1
    min_time_diff = float('inf')

    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, _ = turns[mid]
        

        if s <= tsec <= e:
            return mid # Direct hit

        # Check if tsec is close to this turn
        if tsec < s:
            diff = s - tsec
            if diff < min_time_diff:
                min_time_diff = diff
                best_match_idx = mid
            hi = mid - 1
        else: # tsec > e
            diff = tsec - e
            if diff < min_time_diff:
                min_time_diff = diff
                best_match_idx = mid
            lo = mid + 1
    
    # After binary search, check if the best_match_idx is within tolerance
    if best_match_idx != -1 and min_time_diff <= tolerance:
        return best_match_idx

    return -1


# ============================
#      PIPELINE PRINCIPAL
# ============================

def crop_follow_face_1080x1920_yolo(
    input_path: str,
    output_path: str,
    speech_json: Optional[str] = None,
    static_per_speaker: bool = False,
    highlight_start_sec: float = 0.0,
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

    turns = _load_turns(speech_json, fps, highlight_start_sec) if static_per_speaker and speech_json else []
    use_static = static_per_speaker and len(turns) > 0

    detector = _pick_detector()

    frame_idx = 0
    last_face = None

    # --- Estado del modo dinámico (fallback) ---
    stabilizer = None

    # --- Estado del modo estático por turno ---
    current_turn = -1
    # Valores finales del ancla para el turno actual
    target_anchor_cx: Optional[float] = None
    target_anchor_cy: Optional[float] = None
    target_anchor_win_w: Optional[int] = None
    # Valores de la rampa de transición
    static_ramp_frames_left = 0
    static_ramp_total_frames = 0
    start_anchor_cx, start_anchor_cy, start_anchor_win_w = 0.0, 0.0, 0
    # Valores a usar en el frame actual (pueden estar en plena rampa)
    current_cx, current_cy, current_win_w = 0.0, 0.0, 0

    prev_gray = None
    cut_cooldown = 0
    t0 = time.time()

    last_valid_face_anchor = None # Initialize new variable

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
            current_cy, start_anchor_cy, target_anchor_cy = H / 2.0, H / 2.0, H / 2.0
            current_win_w, start_anchor_win_w, target_anchor_win_w = int(OUT_W * 1.20), int(OUT_W * 1.20), int(OUT_W * 1.20)


        face = detector.detect(frame)
        detected = face is not None

        if detected:
            x, y, w, h = face
            x, y, w, h = (int(round(c * scale)) for c in (x, y, w, h))
            x, y, w, h = _expand_bbox(x, y, w, h, FACE_SCALE_BOX, W, H)
            last_face = (x, y, w, h)

        tsec = frame_idx / float(fps)
        gray = cv2.cvtColor(frs, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            pass # Cut detection logic removed for now
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
                start_anchor_cy = current_cy
                start_anchor_win_w = current_win_w
                
                max_win = min(int(OUT_W * ZOOM_MAX_WIN_MULT), W)
                
                if current_turn >= 0:
                    if last_face is not None:
                        fx, fy, fw, fh = last_face
                        target_anchor_cx = fx + fw / 2.0
                        target_anchor_cy = fy + fh / 2.0
                        target_w = int(round(max(fw / max(FACE_TARGET_RATIO, 1e-4), int(OUT_W * 1.15))))
                        target_anchor_win_w = int(np.clip(target_w, OUT_W, max_win))
                        # Update last_valid_face_anchor
                        last_valid_face_anchor = (target_anchor_cx, target_anchor_cy, target_anchor_win_w)
                    elif last_valid_face_anchor is not None:
                        # If no face detected, but we have a last valid anchor, use it
                        target_anchor_cx, target_anchor_cy, target_anchor_win_w = last_valid_face_anchor
                    else: # Fallback to center if no face and no valid anchor
                        target_anchor_cx = W / 2.0
                        target_anchor_cy = H / 2.0
                        target_anchor_win_w = int(np.clip(int(OUT_W * 1.20), OUT_W, max_win))
                else: # If no turn, general shot (center)
                    target_anchor_cx = W / 2.0
                    target_anchor_cy = H / 2.0
                    target_anchor_win_w = int(np.clip(int(OUT_W * 1.20), OUT_W, max_win))

                
                static_ramp_frames_left = static_ramp_total_frames

            # --- Aplicar rampa de transición si está activa ---
            if static_ramp_frames_left > 0:
                t = 1.0 - (static_ramp_frames_left / float(static_ramp_total_frames))
                t2 = 3*t*t - 2*t*t*t  # Ease-in-out
                current_cx = (1.0 - t2) * start_anchor_cx + t2 * target_anchor_cx
                current_cy = (1.0 - t2) * start_anchor_cy + t2 * target_anchor_cy
                current_win_w = (1.0 - t2) * start_anchor_win_w + t2 * target_anchor_win_w
                static_ramp_frames_left -= 1
            else:
                current_cx = target_anchor_cx
                current_cy = target_anchor_cy
                current_win_w = target_anchor_win_w

            # Aplicar recorte estático (o en transición)
            # Calcular el tamaño ideal del recorte manteniendo el aspecto 9:16
            target_crop_width = current_win_w
            target_crop_height = target_crop_width / TARGET_ASPECT_RATIO

            # Ajustar el tamaño del recorte si excede los límites del frame original
            scale_factor = 1.0
            if target_crop_width > W:
                scale_factor = min(scale_factor, W / target_crop_width)
            if target_crop_height > H:
                scale_factor = min(scale_factor, H / target_crop_height)
            
            crop_width = int(round(target_crop_width * scale_factor))
            crop_height = int(round(target_crop_height * scale_factor))

            # Asegurar que las dimensiones mínimas sean al menos OUT_W y OUT_H (si es posible)
            # Esto es para evitar que el crop sea demasiado pequeño y se vea pixelado
            if crop_width < OUT_W:
                scale_factor = OUT_W / crop_width
                crop_width = OUT_W
                crop_height = int(round(crop_height * scale_factor))
            if crop_height < OUT_H:
                scale_factor = OUT_H / crop_height
                crop_height = OUT_H
                crop_width = int(round(crop_width * scale_factor))

            # Recalcular el centro si las dimensiones se ajustaron
            # current_cx, current_cy ya están suavizados por la rampa

            # Calcular las coordenadas iniciales del recorte
            left = int(round(current_cx - crop_width / 2.0))
            top = int(round(current_cy - crop_height / 2.0))

            # Asegurar que el recorte no se salga de los bordes del frame original
            left = max(0, min(W - crop_width, left))
            top = max(0, min(H - crop_height, top))
            right = left + crop_width
            bottom = top + crop_height

            win = frs[top:bottom, left:right]
            
            crop = cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            writer.write(crop)

        # ------------------
        # MODO DINÁMICO
        # ------------------
        else:
            if stabilizer is None:
                stabilizer = OneEuroStabilizer(freq=fps, frame_width=W, frame_height=H)

            # Actualizar el estabilizador y obtener los valores suavizados
            current_cx, current_cy, current_win_w = stabilizer.update(last_face, tsec)

            # Aplicar recorte dinámico
            # Calcular el tamaño ideal del recorte manteniendo el aspecto 9:16
            target_crop_width = current_win_w
            target_crop_height = target_crop_width / TARGET_ASPECT_RATIO

            # Ajustar el tamaño del recorte si excede los límites del frame original
            scale_factor = 1.0
            if target_crop_width > W:
                scale_factor = min(scale_factor, W / target_crop_width)
            if target_crop_height > H:
                scale_factor = min(scale_factor, H / target_crop_height)
            
            crop_width = int(round(target_crop_width * scale_factor))
            crop_height = int(round(target_crop_height * scale_factor))

            # Asegurar que las dimensiones mínimas sean al menos OUT_W y OUT_H (si es posible)
            # Esto es para evitar que el crop sea demasiado pequeño y se vea pixelado
            if crop_width < OUT_W:
                scale_factor = OUT_W / crop_width
                crop_width = OUT_W
                crop_height = int(round(crop_height * scale_factor))
            if crop_height < OUT_H:
                scale_factor = OUT_H / crop_height
                crop_height = OUT_H
                crop_width = int(round(crop_width * scale_factor))

            # Recalcular el centro si las dimensiones se ajustaron
            # current_cx, current_cy ya están suavizados por el estabilizador

            # Calcular las coordenadas iniciales del recorte
            left = int(round(current_cx - crop_width / 2.0))
            top = int(round(current_cy - crop_height / 2.0))

            # Asegurar que el recorte no se salga de los bordes del frame original
            left = max(0, min(W - crop_width, left))
            top = max(0, min(H - crop_height, top))
            right = left + crop_width
            bottom = top + crop_height

            win = frs[top:bottom, left:right]
            
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
