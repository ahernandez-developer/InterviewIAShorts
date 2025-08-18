# main.py
import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extract_audio_wav, trim_video_ffmpeg
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
# En lugar del FaceCrop “clásico”, usa el YOLO/DNN:
from Components.FaceCropYOLO import crop_follow_face_1080x1920_yolo
from Components.FaceCropYOLO import mux_audio_video_nvenc
from Components.Subtitles import generate_ass, burn_in_subtitles

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

def hr(title: str):
    print("\n" + "="*12 + f" {title} " + "="*12)

class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
    def end(self):
        dt = time.time() - self.t0
        print(f"⏱️  {self.name} completado en {dt:.1f}s")

def build_transcript_string(transcriptions: List[Dict[str, Any]]):
    chunks = []
    for seg in transcriptions:
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        text = seg.get('text', '')
        chunks.append(f"{start:.2f} - {end:.2f}: {text}\n")
    return "".join(chunks)

def guess_companion_audio(final_mp4: Path) -> Path | None:
    folder = final_mp4.parent
    cands = sorted(list(folder.glob("audio_*.* Willow")))
    return cands[0] if cands else None

def make_project_dirs(final_mp4: Path) -> tuple[Path, Path]:
    base = final_mp4.stem
    wdir = WORK / base
    odir = OUT / base
    wdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    return wdir, odir

def main():
    try:
        hr("Descarga")
        url = input("Enter YouTube video URL: ").strip()
        st = StepTimer("Descarga y merge")
        final_path_str = download_youtube_video(url)
        st.end()

        if not final_path_str or not Path(final_path_str).exists():
            logging.error("Unable to Download the video")
            return

        final_mp4 = Path(final_path_str)
        wdir, odir = make_project_dirs(final_mp4)
        logging.info(f"Work dir: {wdir}")
        logging.info(f"Out  dir: {odir}")

        hr("Audio → WAV 16k")
        st = StepTimer("Extracción de audio")
        companion_audio = guess_companion_audio(final_mp4)
        audio_src = companion_audio if companion_audio else final_mp4
        wav_path = wdir / "audio.wav"
        extract_audio_wav(src=str(audio_src), wav=str(wav_path))
        st.end()
        if not wav_path.exists():
            logging.error("No audio file found for ASR")
            return
        logging.info(f"Audio for ASR: {wav_path}")

        hr("Transcripción (ASR) con Word Timestamps")
        st = StepTimer("Transcripción")
        speech_json_path = wdir / "speech.json"

        # TranscribeAudio ahora devuelve una lista de dicts con word-level timestamps
        transcriptions = transcribeAudio(
            str(wav_path),
            model_size="medium",
            language="es",
            beam_size=1,
            vad_filter=True,
            diarization="auto",
            write_speech_json_to=str(speech_json_path),
        )
        st.end()
        if not transcriptions:
            logging.error("Transcription returned empty result")
            return
        
        # El JSON ya se guarda dentro de transcribeAudio, no es necesario hacerlo aquí.

        hr("Selección de highlight (LLM)")
        trans_text = build_transcript_string(transcriptions)
        start_sec, end_sec = GetHighlight(trans_text)
        if not (isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)) and end_sec > start_sec):
            logging.error(f"Invalid highlight window: start={start_sec}, end={end_sec}")
            return
        print(f"🎯 Highlight: {start_sec:.2f}s → {end_sec:.2f}s")

        hr("Recorte (Trim)")
        st = StepTimer("Recorte")
        cut_path = wdir / "cut.mp4"
        # Forzar siempre el re-encode para asegurar cortes precisos y evitar desincronización de subtítulos.
        # El método de copia de stream (-c copy) puede ser impreciso si el tiempo de inicio no es un keyframe.
        trim_video_ffmpeg(
            src=str(final_mp4),
            dst=str(cut_path),
            start=float(start_sec),
            end=float(end_sec),
            copy=False  # <-- Forzar re-encode
        )
        logging.info("Temporal cut exported with re-encode for frame accuracy.")
        st.end()

        hr("Crop 9:16 + seguimiento")
        st = StepTimer("Crop")
        cropped_path = wdir / "cropped.mp4"
        crop_follow_face_1080x1920_yolo(
            input_path=str(cut_path),
            output_path=str(cropped_path),
            speech_json=str(speech_json_path),
            static_per_speaker=True
        )
        st.end()
        logging.info(f"Cropped clip: {cropped_path}")

        hr("Mux final (NVENC)")
        st = StepTimer("Mux")
        final_short = odir / "Final.mp4"
        mux_audio_video_nvenc(
            video_with_audio=str(cut_path),
            video_without_audio=str(cropped_path),
            dst=str(final_short),
            fps=30,
            v_bitrate="6M"
        )
        st.end()
        print(f"✅ Short (sin subtítulos) listo en: {final_short}")

        hr("Generación de Subtítulos Dinámicos")
        st = StepTimer("Generación de ASS y Burn-in")

        # Filtrar los segmentos de la transcripción que caen en el highlight
        highlight_segments = []
        for segment in transcriptions:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            if max(seg_start, start_sec) < min(seg_end, end_sec):
                # Ajustar timestamps de los segmentos y sus palabras al inicio del clip
                new_seg = segment.copy()
                new_seg['start'] = seg_start - start_sec
                new_seg['end'] = seg_end - start_sec
                
                new_words = []
                if 'words' in new_seg and new_seg['words'] is not None:
                    for word_info in new_seg['words']:
                        new_word_info = word_info.copy()
                        new_word_info['start'] = word_info['start'] - start_sec
                        new_word_info['end'] = word_info['end'] - start_sec
                        new_words.append(new_word_info)
                new_seg['words'] = new_words
                highlight_segments.append(new_seg)

        ass_path = odir / "subtitles.ass"
        generate_ass(highlight_segments, ass_path)

        final_subtitled_short = odir / f"{final_short.stem}_subtitled.mp4"
        burn_in_subtitles(
            video_path=final_short,
            subtitle_path=ass_path,
            output_path=final_subtitled_short
        )
        st.end()

        print(f"\n✅ Done! Short final con subtítulos dinámicos en: {final_subtitled_short}\n")

    except KeyboardInterrupt:

        print("\nInterrupted by user.")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
