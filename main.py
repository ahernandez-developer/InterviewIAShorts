# main.py (pipeline optimizado: trim primero, luego escala/crop)
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# === Componentes propios ===
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import (
    extract_audio_wav,         # ffmpeg → WAV 16k mono (desde .m4a/.webm/.mp4)
    trim_video_ffmpeg,         # recorte rápido (fast-seek) y opción de -c copy
)
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import (
    crop_follow_face_1080x1920,  # crop vertical estable (1080x1920) con seguimiento en X
    mux_audio_video_nvenc,        # reinyecta audio del recorte al video cropeado (NVENC)
)

# ========== Configuración ==========
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROOT = Path(__file__).parent
WORK = ROOT / "work"
OUT  = ROOT / "out"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


# ========= Utilidades =========
def build_transcript_string(transcriptions):
    """
    Convierte [[text, start, end], ...] en un string lineal (conserva timestamps)
    para alimentar el selector de highlight.
    """
    chunks = []
    for text, start, end in transcriptions:
        chunks.append(f"{start} - {end}: {text}\n")
    return "".join(chunks)

def guess_companion_audio(final_mp4: Path) -> Path | None:
    """
    Intenta ubicar el audio descargado (audio_*.m4a o audio_*.webm) dentro de la
    misma carpeta del video para acelerar el ASR (evita decodificar desde el MP4).
    """
    folder = final_mp4.parent
    cands = sorted(list(folder.glob("audio_*.*")))
    return cands[0] if cands else None

def make_project_dirs(final_mp4: Path) -> tuple[Path, Path]:
    """
    Crea subcarpetas work/out específicas para el video:
      work/<basename>/, out/<basename>/
    """
    base = final_mp4.stem  # p.ej. 25_08_17_nodal_rompe_el_silencio
    wdir = WORK / base
    odir = OUT / base
    wdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    return wdir, odir


# ========= Flujo principal =========
def main():
    try:
        # 1) Ingesta
        url = input("Enter YouTube video URL: ").strip()
        final_path_str = download_youtube_video(url)  # retorna .../videos/<base>/<base>.mp4
        if not final_path_str or not Path(final_path_str).exists():
            logging.error("Unable to Download the video")
            return

        final_mp4 = Path(final_path_str)
        logging.info(f"Downloaded video at: {final_mp4}")

        # Subcarpetas dedicadas a este video
        wdir, odir = make_project_dirs(final_mp4)
        logging.info(f"Work dir: {wdir}")
        logging.info(f"Out  dir: {odir}")

        # 2) Extraer audio para ASR (preferir el audio descargado si existe)
        companion_audio = guess_companion_audio(final_mp4)
        audio_src = companion_audio if companion_audio else final_mp4
        wav_path = wdir / "audio.wav"
        extract_audio_wav(src=str(audio_src), wav=str(wav_path))
        if not wav_path.exists():
            logging.error("No audio file found for ASR")
            return
        logging.info(f"Audio for ASR: {wav_path}")

        # 3) Transcripción (GPU si disponible; fallback CPU si falta cuDNN)
        transcriptions = transcribeAudio(str(wav_path))
        if not transcriptions:
            logging.error("Transcription returned empty result")
            return

        # 4) LLM → highlight (start, end) en segundos
        trans_text = build_transcript_string(transcriptions)
        start_sec, end_sec = GetHighlight(trans_text)
        if not (isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)) and end_sec > start_sec):
            logging.error(f"Invalid highlight window: start={start_sec}, end={end_sec}")
            return
        logging.info(f"Highlight → start={start_sec:.3f}s, end={end_sec:.3f}s")

        # 5) Recortar PRIMERO del original (más rápido). Intentar -c copy si solo queremos aislar.
        #    Nota: si luego vamos a escalar/cropear con OpenCV, igual habrá un re-encode. Aun así,
        #          recortar primero ahorra muchísimo tiempo.
        cut_path = wdir / "cut.mp4"
        # Intento 1: fast-seek + copy (si contenedor/codec lo permite). Si falla, el helper lanza y capturamos para caer al encode.
        try:
            trim_video_ffmpeg(
                src=str(final_mp4),
                dst=str(cut_path),
                start=float(start_sec),
                end=float(end_sec),
                copy=True  # prueba copy primero (rápido)
            )
            logging.info("Temporal cut exported with -c copy (fast).")
        except Exception as e:
            logging.warning(f"Fast copy trim failed ({e}). Retrying with re-encode.")
            trim_video_ffmpeg(
                src=str(final_mp4),
                dst=str(cut_path),
                start=float(start_sec),
                end=float(end_sec),
                copy=False  # recorta re-encode (NVENC/libx264 según disponibilidad)
            )
            logging.info("Temporal cut exported with re-encode.")

        # 6) Crop vertical 1080x1920 con seguimiento horizontal
        cropped_path = wdir / "cropped.mp4"
        crop_follow_face_1080x1920(
            input_path=str(cut_path),
            output_path=str(cropped_path)
        )
        logging.info(f"Cropped clip: {cropped_path}")

        # 7) Reinyectar audio del recorte al video cropeado (NVENC)
        final_short = odir / "Final.mp4"
        mux_audio_video_nvenc(
            video_with_audio=str(cut_path),
            video_without_audio=str(cropped_path),
            dst=str(final_short),
            fps=30,
            v_bitrate="6M"  # 5–6M suele ser suficiente para talking-head
        )

        print(f"\n✅ Done! Short ready at: {final_short}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
