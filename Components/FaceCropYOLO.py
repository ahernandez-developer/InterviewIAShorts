# Components/FaceCropYOLO.py
from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import cv2
import torch

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

from .SystemMonitor import SystemMonitor
from OneEuroFilter import OneEuroFilter
from .common_ffmpeg import run_ffmpeg_with_progress

# ... (El resto de los PARÁMETROS GLOBALES, HELPERS, etc. se mantienen igual)

# ============================
#     PARÁMETROS GLOBALES
# ============================

# Salida vertical 9:16
OUT_W, OUT_H = 1080, 1920
TARGET_ASPECT_RATIO = OUT_W / OUT_H

# Detección
CONF_THR = 0.35
FACE_SCALE_BOX = 1.50

# --- Parámetros de la Cámara de Muelle (Spring Camera) ---
POS_STIFFNESS = 0.08
POS_DAMPING = 0.60
ZOOM_STIFFNESS = 0.05
ZOOM_DAMPING = 0.75
FACE_TARGET_RATIO = 0.28

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

# ============================
#         DETECTORES
# ============================

class _YoloFaceDetector:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"

    def detect_video(self, video_path: str, progress: "Progress | None" = None, task_id: "TaskID | None" = None) -> List[Optional[Tuple[int, int, int, int]]]:
        results_generator = self.model.predict(
            source=video_path, stream=True, verbose=False, conf=CONF_THR,
            device=self.device, half=self.half
        )
        detections = []
        prev_cx = None
        
        # Get total frames for progress bar
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if progress and task_id is not None:
            progress.update(task_id, total=total_frames)
            iterator = results_generator
        else:
            iterator = tqdm(results_generator, desc="[Pasada 1] Detectando caras")

        for frame_idx, results in enumerate(iterator):
            if progress and task_id is not None:
                progress.update(task_id, completed=frame_idx + 1)

            if results.boxes is None or len(results.boxes) == 0:
                detections.append(None)
                continue
            boxes = results.boxes.xywh.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            if prev_cx is not None:
                centers_x = boxes[:, 0]
                idx = np.argmin(np.abs(centers_x - prev_cx))
            else:
                idx = np.argmax(confs)
            best_box = boxes[idx]
            x, y, w, h = [int(v) for v in best_box]
            detections.append((x, y, w, h))
            prev_cx = x
        return detections

# ... (Clase _Res10DnnDetector y _pick_detector se mantienen igual)
class _Res10DnnDetector:
    pass # Mantener la implementación original

def _pick_detector():
    weights = os.getenv("FACE_MODEL_PATH", "models/yolo/yolov8n-face-lindevs.pt")
    if Path(weights).exists():
        print(f"[Face] Using YOLO weights: {weights}")
        try:
            return _YoloFaceDetector(weights)
        except Exception as e:
            print(f"[Face] YOLO init failed ({e}), fallback to DNN...")
    # Lógica de fallback a DNN
    proto = Path("deploy.prototxt")
    cafe  = Path("res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if not proto.exists() or not cafe.exists():
        raise FileNotFoundError("Modelo YOLO no encontrado y archivos de fallback DNN tampoco.")
    print(f"[Face] Using DNN Res10: {proto.name}, {cafe.name}")
    return _Res10DnnDetector(str(proto), str(cafe))


# ============================
#     LÓGICA DE CÁMARA
# ============================

class CameraSpring:
    # ... (sin cambios)
    def __init__(self, stiffness: float = 0.1, damping: float = 0.5):
        self.stiffness = stiffness
        self.damping = damping
        self.pos = 0.0
        self.vel = 0.0

    def reset_to(self, target: float):
        self.pos = target
        self.vel = 0.0

    def update(self, target: float, dt: float = 1.0) -> float:
        spring_force = (target - self.pos) * self.stiffness
        damping_force = -self.vel * self.damping
        acceleration = (spring_force + damping_force) * dt
        self.vel += acceleration
        self.pos += self.vel * dt
        return self.pos

def _load_turns(speech_json: str | Path, fps: float, highlight_start_sec: float) -> List[Tuple[float, float, str]]:
    # ... (sin cambios)
    import json
    p = Path(speech_json)
    if not p.exists(): return []
    with open(p, "r", encoding="utf-8") as f: items = json.load(f)
    items.sort(key=lambda d: float(d.get("start", 0.0)))
    turns: List[Tuple[float, float, str]] = []
    cur_spk, cur_s, cur_e = None, None, None
    def _push():
        if cur_spk is None: return
        if (cur_e - cur_s) >= 0.8: turns.append((float(cur_s), float(cur_e), str(cur_spk)))
    for it in items:
        spk, s, e = str(it.get("speaker", "SPEAKER_0")), float(it.get("start", 0.0)) - highlight_start_sec, float(it.get("end", 0.0)) - highlight_start_sec
        s, e = max(0.0, s), max(0.0, e)
        if cur_spk is None: cur_spk, cur_s, cur_e = spk, s, e; continue
        if spk == cur_spk and s <= (cur_e + 0.25): cur_e = max(cur_e, e)
        else: _push(); cur_spk, cur_s, cur_e = spk, s, e
    _push()
    return turns

def _find_turn_index(turns: List[Tuple[float,float,str]], tsec: float) -> int:
    # ... (sin cambios)
    lo, hi = 0, len(turns)-1
    tolerance = 0.15
    best_match_idx, min_time_diff = -1, float('inf')
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, _ = turns[mid]
        if s <= tsec <= e: return mid
        diff = s - tsec if tsec < s else tsec - e
        if diff < min_time_diff: min_time_diff, best_match_idx = diff, mid
        if tsec < s: hi = mid - 1
        else: lo = mid + 1
    return best_match_idx if min_time_diff <= tolerance else -1

# ============================
#      PIPELINE PRINCIPAL
# ============================

def crop_follow_face_1080x1920_yolo(
    input_path: str,
    output_path: str,
    speech_json: Optional[str] = None,
    static_per_speaker: bool = False,
    highlight_start_sec: float = 0.0,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None
):
    detector = _pick_detector()
    if not isinstance(detector, _YoloFaceDetector):
        raise NotImplementedError("El modo de dos pasadas solo está implementado para YOLO.")

    monitor = SystemMonitor()

    # --- PASADA 1: DETECCIÓN ---
    monitor.start()
    detection_task_id = None
    if progress and task_id is not None:
        detection_task_id = progress.add_task("[yellow]Detectando caras...[/yellow]", total=1, parent=task_id, visible=True)
    all_face_detections = detector.detect_video(input_path, progress=progress, task_id=detection_task_id)
    if progress and detection_task_id is not None: progress.update(detection_task_id, completed=100, description="[green]Detección de caras completada.[/green]")
    pass1_stats = monitor.stop()

    # --- PASADA 2: CÁMARA Y RENDER ---
    monitor.start()
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer_with_fallback(output_path, fps)

    turns = _load_turns(speech_json, fps, highlight_start_sec) if static_per_speaker and speech_json else []
    use_static = static_per_speaker and len(turns) > 0

    cam_x, cam_y, cam_zoom = CameraSpring(POS_STIFFNESS, POS_DAMPING), CameraSpring(POS_STIFFNESS, POS_DAMPING), CameraSpring(ZOOM_STIFFNESS, ZOOM_DAMPING)
    target_anchor_cx, target_anchor_cy, target_anchor_win_w = W / 2.0, H / 2.0, W * 0.8
    cam_x.reset_to(target_anchor_cx); cam_y.reset_to(target_anchor_cy); cam_zoom.reset_to(target_anchor_win_w)

    # --- Archivo de Debug para Movimiento de Cámara ---
    debug_log_path = Path(output_path).parent / "camera_movement.csv"
    debug_log_file = open(debug_log_path, "w")
    debug_log_file.write("frame,target_x,current_x,target_y,current_y,target_zoom,current_zoom\n")
    # ----------------------------------------------------

    current_turn, is_searching, search_frames = -1, False, 0

    render_task_id = None
    if progress and task_id is not None:
        render_task_id = progress.add_task("[cyan]Renderizando video...[/cyan]", total=len(all_face_detections), parent=task_id, visible=True)

    for frame_idx, face_box in enumerate(all_face_detections):
        if progress and render_task_id is not None:
            progress.update(render_task_id, completed=frame_idx + 1)

        ok, frame = cap.read()
        if not ok: break

        if face_box: face_box = _expand_bbox(*face_box, FACE_SCALE_BOX, W, H)

        tsec = frame_idx / fps

        if use_static:
            idx = _find_turn_index(turns, tsec)
            if idx != current_turn: is_searching, search_frames, current_turn = True, 0, idx
            if is_searching:
                search_frames += 1
                if face_box:
                    fx, fy, fw, fh = face_box
                    target_anchor_cx, target_anchor_cy, target_anchor_win_w = fx + fw / 2.0, fy + fh / 2.0, fw / FACE_TARGET_RATIO
                    is_searching = False
                elif search_frames > (fps * 1.5):
                    target_anchor_cx, target_anchor_cy, target_anchor_win_w = W / 2.0, H / 2.0, W * 0.8
                    is_searching = False
        else: # Modo dinámico
            if face_box:
                fx, fy, fw, fh = face_box
                target_anchor_cx, target_anchor_cy, target_anchor_win_w = fx + fw / 2.0, fy + fh / 2.0, fw / FACE_TARGET_RATIO

        current_cx, current_cy, current_win_w = cam_x.update(target_anchor_cx), cam_y.update(target_anchor_cy), cam_zoom.update(target_anchor_win_w)

        # --- Guardar datos de debug ---
        debug_log_file.write(
            f"{frame_idx},{target_anchor_cx:.2f},{current_cx:.2f},{target_anchor_cy:.2f},{current_cy:.2f},{target_anchor_win_w:.2f},{current_win_w:.2f}\n"
        )
        # -----------------------------

        crop_w = current_win_w
        crop_h = crop_w / TARGET_ASPECT_RATIO
        if crop_w > W: crop_w, crop_h = W, W / TARGET_ASPECT_RATIO
        if crop_h > H: crop_h, crop_w = H, H * TARGET_ASPECT_RATIO
        crop_w, crop_h = int(round(crop_w)), int(round(crop_h))

        left, top = int(round(current_cx - crop_w / 2.0)), int(round(current_cy - crop_h / 2.0))
        left, top = max(0, min(W - crop_w, left)), max(0, min(H - crop_h, top))

        win = frame[top:top+crop_h, left:left+crop_w]
        final_crop = cv2.resize(win, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        writer.write(final_crop)

    debug_log_file.close()
    writer.release()
    cap.release()
    pass2_stats = monitor.stop()


def mux_audio_video(video_with_audio: str, video_without_audio: str, dst: str,
                      fps: int = 30,
                      progress: "Progress | None" = None,
                      task_id: "TaskID | None" = None):
    ff = shutil.which("ffmpeg.exe" if str(Path(os.sys.executable)).lower().startswith("c:") else "ffmpeg")
    if not ff: raise RuntimeError("FFmpeg no encontrado")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    # Get total duration of the video_without_audio for progress bar
    cap = cv2.VideoCapture(video_without_audio)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / (cap.get(cv2.CAP_PROP_FPS) or fps)
    cap.release()

    cmd = [ff, "-y", "-i", video_without_audio, "-i", video_with_audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-r", str(fps), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", dst]
    run_ffmpeg_with_progress(cmd, total_duration=video_duration, label="Muxing", progress=progress, task_id=task_id)
