# Components/FaceCropYOLO.py
from __future__ import annotations
import os, json, time, shutil
from pathlib import Path
from collections import deque
import numpy as np
import cv2

# Salida vertical 9:16
OUT_W, OUT_H = 1080, 1920

# Detección
DET_SIZE = 640
CONF_THR = 0.35
FACE_SCALE_BOX = 1.32     # expandimos bbox para no cortar frente/mentón

# Estabilización (paneo horizontal)
DEADZONE_PX = 12
MAX_STEP_PX = 18
SPRING_K    = 0.10
DAMPING     = 0.88
TARGET_MEDIAN_WIN = 5
CUT_JUMP_THRESH   = 240
CUT_RAMP_FRAMES   = 10
MISS_HYSTERESIS   = 8
EDGE_MARGIN       = 26

# “Cámara estática” (zoom fijo o casi fijo)
STATIC_ZOOM_WIN_W = OUT_W           # ancho deseado de la ventana de recorte
STATIC_ZOOM_ALPHA = 0.08            # si quieres micro-ajustes lentos; 0 = fijo

# Lógica de cambio guiado por voz
SPEECH_DWELL_SEC       = 0.6        # mínimo hablando para aceptar cambio
SILENCE_SWITCH_WINDOW  = 0.35       # idealmente cambiamos en esta pausa (s)
FACE_MARGIN_IN_CROP_PX = 32         # margen mínimo de cara al borde del crop

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

# ---------- Detectores ----------
class _YoloFaceDetector:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.half = self.device != "cpu"

    def detect_many(self, frame_bgr):
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
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), c in zip(boxes, confs):
            # deshacer padding + escala
            x1 = (x1 - pad_left) / scale; x2 = (x2 - pad_left) / scale
            y1 = (y1 - pad_top) / scale;  y2 = (y2 - pad_top) / scale
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
        return out

class _Res10DnnDetector:
    def __init__(self, prototxt: str, caffemodel: str, conf_thr: float = 0.5):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf_thr = conf_thr

    def detect_many(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        out = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.conf_thr:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
            out.append((x1, y1, x2-x1, y2-y1, score))
        return out

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

# ---------- Estabilizador ----------
class CinematicStabilizer:
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
            t2 = 3*t*t - 2*t*t*t
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

# ---------- VAD / Speech gating ----------
class SpeechGate:
    """Gestiona 'hay voz' y ventanas ideales para cambiar de sujeto."""
    def __init__(self, fps: float, speech_json: str | None = None):
        self.fps = fps
        self.t = 0.0
        self.idx = 0
        self.speech = []  # lista de (start,end)
        if speech_json and Path(speech_json).exists():
            try:
                data = json.loads(Path(speech_json).read_text(encoding="utf-8"))
                for seg in data:
                    self.speech.append((float(seg["start"]), float(seg["end"])))
            except Exception:
                pass
        self.in_speech_prev = False
        self.t_started = None

    def _simple_vad(self, mono16: np.ndarray | None):
        # Placeholder: retornamos estado previo si no hay VAD externo.
        # Como llamamos por frame, usaremos las ventanas de self.speech si existen.
        return None

    def step(self):
        """Avanza 1 frame en el timeline, retorna: in_speech, can_switch_now"""
        t = self.t
        in_speech = False
        if self.speech:
            # ¿t está dentro de alguno?
            for s, e in self.speech:
                if s <= t <= e:
                    in_speech = True
                    break

        # control de dwell y “switch en silencio”
        can_switch = False
        if in_speech:
            # acumulamos dwell
            if not self.in_speech_prev:
                self.t_started = t
            elif self.t_started is not None and (t - self.t_started) >= SPEECH_DWELL_SEC:
                # ya cumplimos dwell, pero intentaremos cambiar al entrar a silencio
                can_switch = False
        else:
            # silencio: si veníamos de voz y teníamos dwell suficiente, permitir switch
            if self.in_speech_prev and self.t_started is not None:
                if (t - self.t_started) >= SPEECH_DWELL_SEC:
                    can_switch = True
                self.t_started = None

        self.in_speech_prev = in_speech
        self.t += 1.0 / max(self.fps, 1e-6)
        return in_speech, can_switch

# ---------- Pipeline principal ----------
def crop_follow_face_1080x1920_yolo(
    input_path: str,
    output_path: str,
    speech_json: str | None = None,   # opcional: segmentos de voz [{"start":s,"end":e},...]
):
    """
    Detección por frame (YOLO o DNN) + estabilización + 'cámara estática' (zoom fijo)
    con gating por voz: solo cambiamos de objetivo en pausas y tras un mínimo hablando.
    """
    _ensure_parent(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = _open_writer_with_fallback(output_path, fps)

    detector = _pick_detector()
    frame_idx = 0
    t0 = time.time()

    last_face = None
    stabilizer = None

    # estado de objetivo (persona/cara elegida)
    target_face = None        # (x,y,w,h) en espacio reescalado
    frames_on_target = 0

    # ventana de recorte (zoom estático con micro-ajuste)
    win_w = STATIC_ZOOM_WIN_W

    gate = SpeechGate(fps=fps, speech_json=speech_json)

    def _choose_face(faces_rs, current):
        """Selecciona cara según (1) IOU con actual, (2) tamaño, (3) centrado."""
        if not faces_rs:
            return None
        if current is None:
            # la más grande
            return max(faces_rs, key=lambda b: b[2]*b[3])[:4]
        # mejor IOU con la actual, luego tamaño
        cx, cy, cw, ch = current
        best, best_key = None, (-1.0, -1.0)
        for (x,y,w,h,_) in faces_rs:
            inter_x1 = max(x, cx); inter_y1 = max(y, cy)
            inter_x2 = min(x+w, cx+cw); inter_y2 = min(y+h, cy+ch)
            inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
            iou = inter / max(1.0, (w*h + cw*ch - inter))
            size = w*h
            key = (iou, size)
            if key > best_key:
                best_key, best = key, (x,y,w,h)
        return best

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame is None:
            continue
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frs, W, H, scale = _resize_to_h(frame)
        if stabilizer is None:
            stabilizer = CinematicStabilizer(W)

        # detección (todas las caras)
        faces = detector.detect_many(frame)  # en espacio original
        faces_rs = []
        for (x,y,w,h,c) in faces:
            x = int(round(x*scale)); y = int(round(y*scale))
            w = int(round(w*scale)); h = int(round(h*scale))
            x,y,w,h = _expand_bbox(x,y,w,h, FACE_SCALE_BOX, W, H)
            faces_rs.append((x,y,w,h,c))

        in_speech, can_switch = gate.step()

        # elegir/retener objetivo
        if target_face is None:
            target_face = _choose_face(faces_rs, None)
            frames_on_target = 0
        else:
            # ¿sigue visible la cara objetivo?
            still = None
            for (x,y,w,h,_) in faces_rs:
                # IOU con objetivo actual
                ox,oy,ow,oh = target_face
                inter_x1 = max(x, ox); inter_y1 = max(y, oy)
                inter_x2 = min(x+w, ox+ow); inter_y2 = min(y+h, oy+oh)
                inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
                iou = inter / max(1.0, (w*h + ow*oh - inter))
                if iou > 0.25:
                    still = (x,y,w,h)
                    break

            if still is not None:
                target_face = still
                frames_on_target += 1
            else:
                # la cara se perdió. ¿Podemos cambiar ahora?
                if can_switch:
                    cand = _choose_face(faces_rs, target_face)
                    if cand is not None:
                        target_face = cand
                        frames_on_target = 0
                # si no podemos cambiar, mantenemos el último encuadre (no recentrar).

        # si no hay ninguna cara ahora mismo, solo mantenemos paneo previo
        face_for_frame = target_face

        # paneo suavizado
        if face_for_frame is not None:
            x,y,w,h = face_for_frame
            cx_target = x + w/2.0
            stabilizer.update_target(cx_target, detected=True)
        else:
            stabilizer.update_target(W/2.0, detected=False)

        cx = stabilizer.cx_smoothed

        # ======= cámara “estática”: win_w casi fijo + garantía de cara completa =======
        # micro-ajuste lento hacia STATIC_ZOOM_WIN_W (si lo quieres 100% fijo, usa alpha=0)
        win_w = int(round(STATIC_ZOOM_ALPHA * STATIC_ZOOM_WIN_W + (1-STATIC_ZOOM_ALPHA) * win_w))
        win_w = max(OUT_W, min(W, win_w))
        half = win_w / 2.0
        left = int(round(cx - half))

        # Garantizar que la cara quede dentro del crop con margen
        if face_for_frame is not None:
            fx,fy,fw,fh = face_for_frame
            face_cx = fx + fw/2
            # corremos la ventana si la cara queda cerca del borde
            if face_cx - left < FACE_MARGIN_IN_CROP_PX:
                left = int(face_cx - FACE_MARGIN_IN_CROP_PX)
            if (left + win_w) - face_cx < FACE_MARGIN_IN_CROP_PX:
                left = int(face_cx + FACE_MARGIN_IN_CROP_PX - win_w)

        # límites
        left = max(0, min(W - win_w, left))
        win = frs[:, left:left+win_w]

        crop = cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        crop = np.ascontiguousarray(crop, dtype=np.uint8)
        writer.write(crop)

        if frame_idx % 60 == 0:
            elapsed = round(time.time() - t0, 2)
            print(f"[FaceCrop] f={frame_idx} | {frame_idx/max(elapsed,1e-6):.1f} FPS | W={W} | win_w={win_w} | speech={in_speech} can_switch={can_switch}")

    writer.release()
    cap.release()
    print("[FaceCrop] listo.")

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
