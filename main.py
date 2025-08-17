# main.py (v2, GPU/NVENC-ready, normalización previa y orquestación limpia)

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# === Componentes (los actualizaremos enseguida) ===
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import (
    extract_audio_wav,         # nuevo: ffmpeg a WAV 16k mono
    normalize_video_9x16_base, # nuevo: scale=-2:1920 + yuv420p (+ NVENC)
    trim_video_ffmpeg,         # nuevo: recorte por -ss/-to (NVENC)
)
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import (
    crop_follow_face_1080x1920, # nuevo: crop vertical estable solo en X
    mux_audio_video_nvenc,      # nuevo: combina video sin audio + audio (NVENC)
)

# =============== Config básica ===============
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Carpetas de trabajo
ROOT = Path(__file__).parent
WORK = ROOT / "work"
OUT  = ROOT / "out"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def build_transcript_string(transcriptions):
    """
    Convierte la lista [[text, start, end], ...] en un string lineal
    que conserva timestamps (como tu main original).
    """
    chunks = []
    for text, start, end in transcriptions:
        chunks.append(f"{start} - {end}: {text}")
    return "".join(chunks)

def main():
    try:
        # 1) Ingesta: URL → descarga en videos/
        url = input("Enter YouTube video URL: ").strip()
        src_path = download_youtube_video(url)  # devuelve ruta al MP4 final en videos/
        if not src_path or not Path(src_path).exists():
            logging.error("Unable to Download the video")
            return

        logging.info(f"Downloaded video at: {src_path}")

        # 2) Normalización (clave para evitar alturas impares y jitter)
        #    - Altura fija 1920 (9:16-ready)
        #    - Formato yuv420p (evita problemas de libx264/h264_nvenc)
        normalized = WORK / "normalized.mp4"
        normalize_video_9x16_base(src=str(src_path), dst=str(normalized), fps=30)
        logging.info(f"Normalized master: {normalized}")

        # 3) Extraer audio WAV 16k mono (mejor para ASR/VAD)
        wav_path = WORK / "audio.wav"
        extract_audio_wav(src=str(normalized), wav=str(wav_path))
        if not wav_path.exists():
            logging.error("No audio file found")
            return
        logging.info(f"Audio extracted: {wav_path}")

        # 4) Transcripción (GPU si disponible)
        #    Nota: ajustaremos Transcription.py para usar CUDA + float16/int8
        transcriptions = transcribeAudio(str(wav_path))
        if not transcriptions:
            logging.warning("No transcriptions found")
            return

        # 5) LLM → rango de highlight (por ahora uno; luego soportaremos N cortes)
        trans_text = build_transcript_string(transcriptions)
        start_sec, end_sec = GetHighlight(trans_text)
        if start_sec == 0 and end_sec == 0:
            logging.error("Error in getting highlight")
            return

        logging.info(f"Highlight chosen → Start: {start_sec} s, End: {end_sec} s")

        # 6) Recorte temporal del master normalizado (NVENC)
        out_cut = WORK / "cut.mp4"
        trim_video_ffmpeg(
            src=str(normalized),
            dst=str(out_cut),
            start=float(start_sec),
            end=float(end_sec),
            fps=30
        )
        logging.info(f"Temporal cut exported: {out_cut}")

        # 7) Crop vertical 1080x1920 con seguimiento en X
        #    (evita "Frame size inconsistant", mantiene H=1920 y W=1080)
        cropped = WORK / "cropped.mp4"
        crop_follow_face_1080x1920(
            input_path=str(out_cut),
            output_path=str(cropped)
        )
        logging.info(f"Cropped clip: {cropped}")

        # 8) Combinar audio (del recorte con audio) con el video cropeado (sin audio)
        #    Podemos reutilizar el audio del out_cut; si crop quita el audio, lo reinyectamos aquí.
        final = OUT / "Final.mp4"
        mux_audio_video_nvenc(
            video_with_audio=str(out_cut),
            video_without_audio=str(cropped),
            dst=str(final),
            fps=30,
            v_bitrate="8M"
        )
        logging.info(f"Final short exported: {final}")

        print(f"\n✅ Done! Short ready at: {final}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
