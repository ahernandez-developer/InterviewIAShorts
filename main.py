import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

# Rich for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extract_audio_wav, trim_video_ffmpeg
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCropYOLO import crop_follow_face_1080x1920_yolo
from Components.FaceCropYOLO import mux_audio_video_nvenc
from Components.Subtitles import generate_ass, burn_in_subtitles

load_dotenv()

# --- Rich Console Setup ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger("rich") # Use rich's logger

# --- Helper Functions (adapted for Rich) ---
def hr(title: str):
    console.print(Panel(f"[bold blue]{title}", expand=False, border_style="blue"))

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
    # Added a check for empty list to prevent IndexError
    return cands[0] if cands else None

def make_project_dirs(final_mp4: Path) -> tuple[Path, Path]:
    base = final_mp4.stem
    wdir = WORK / base
    odir = OUT / base
    wdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    return wdir, odir

# --- Main Pipeline ---
ROOT = Path(__file__).parent
WORK = ROOT / "work"
OUT  = ROOT / "out"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def main():
    try:
        hr("Descarga de Video")
        url = console.input("[bold green]Introduce la URL del video de YouTube:[/bold green] ").strip()
        
        with console.status("[bold yellow]Descargando y fusionando video...[/bold yellow]", spinner="dots"):
            final_path_str = download_youtube_video(url)
        
        if not final_path_str or not Path(final_path_str).exists():
            logger.error("[bold red]No se pudo descargar el video.[/bold red]")
            return

        final_mp4 = Path(final_path_str)
        wdir, odir = make_project_dirs(final_mp4)
        console.print(f"✅ Video descargado en: [green]{final_mp4}[/green]")
        logger.info(f"Directorio de trabajo: [cyan]{wdir}[/cyan]")
        logger.info(f"Directorio de salida: [cyan]{odir}[/cyan]")

        hr("Extracción de Audio")
        with console.status("[bold yellow]Extrayendo audio a WAV...[/bold yellow]", spinner="dots"):
            companion_audio = guess_companion_audio(final_mp4)
            audio_src = companion_audio if companion_audio else final_mp4
            wav_path = wdir / "audio.wav"
            extract_audio_wav(src=str(audio_src), wav=str(wav_path))
        
        if not wav_path.exists():
            logger.error("[bold red]No se encontró el archivo de audio para la transcripción.[/bold red]")
            return
        console.print(f"✅ Audio extraído a: [green]{wav_path}[/green]")

        hr("Transcripción (ASR) con Word Timestamps")
        speech_json_path = wdir / "speech.json"
        with console.status("[bold yellow]Transcribiendo audio (esto puede tardar)...[/bold yellow]", spinner="dots"):
            transcriptions = transcribeAudio(
                str(wav_path),
                model_size="medium",
                language="es",
                beam_size=1,
                vad_filter=True,
                diarization="auto",
                write_speech_json_to=str(speech_json_path),
            )
        
        if not transcriptions:
            logger.error("[bold red]La transcripción no devolvió resultados.[/bold red]")
            return
        console.print(f"✅ Transcripción completada y guardada en: [green]{speech_json_path}[/green]")

        hr("Selección de Highlight (LLM)")
        with console.status("[bold yellow]Seleccionando el highlight con IA...[/bold yellow]", spinner="dots"):
            trans_text = build_transcript_string(transcriptions)
            start_sec, end_sec = GetHighlight(trans_text)
        
        if not (isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)) and end_sec > start_sec):
            logger.error(f"[bold red]Ventana de highlight inválida: start={start_sec}, end={end_sec}[/bold red]")
            return
        console.print(f"🎯 Highlight seleccionado: [green]{start_sec:.2f}s → {end_sec:.2f}s[/green]")

        hr("Recorte Preciso del Video")
        cut_path = wdir / "cut.mp4"
        with console.status("[bold yellow]Recortando el video (re-codificando para precisión)...[/bold yellow]", spinner="dots"):
            trim_video_ffmpeg(
                src=str(final_mp4),
                dst=str(cut_path),
                start=float(start_sec),
                end=float(end_sec),
                copy=False
            )
        console.print(f"✅ Video recortado en: [green]{cut_path}[/green]")

        hr("Cámara Virtual (Crop 9:16 + Seguimiento)")
        cropped_path = wdir / "cropped.mp4"
        with console.status("[bold yellow]Aplicando cámara virtual y recorte 9:16...[/bold yellow]", spinner="dots"):
            crop_follow_face_1080x1920_yolo(
                input_path=str(cut_path),
                output_path=str(cropped_path),
                speech_json=str(speech_json_path),
                static_per_speaker=True
            )
        console.print(f"✅ Clip recortado con cámara virtual: [green]{cropped_path}[/green]")

        hr("Muxing Final (NVENC)")
        final_short = odir / "Final.mp4"
        with console.status("[bold yellow]Fusionando video y audio...[/bold yellow]", spinner="dots"):
            mux_audio_video_nvenc(
                video_with_audio=str(cut_path),
                video_without_audio=str(cropped_path),
                dst=str(final_short),
                fps=30,
                v_bitrate="6M"
            )
        console.print(f"✅ Short (sin subtítulos) listo en: [green]{final_short}[/green]")

        hr("Generación de Subtítulos Dinámicos")
        ass_path = odir / "subtitles.ass"
        final_subtitled_short = odir / f"{final_short.stem}_subtitled.mp4"
        
        with console.status("[bold yellow]Generando subtítulos ASS y quemándolos en el video...[/bold yellow]", spinner="dots"):
            # Filtrar los segmentos de la transcripción que caen en el highlight
            highlight_segments = []
            for segment in transcriptions:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                if max(seg_start, start_sec) < min(seg_end, end_sec):
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

            generate_ass(highlight_segments, ass_path)
            burn_in_subtitles(
                video_path=final_short,
                subtitle_path=ass_path,
                output_path=final_subtitled_short
            )
        console.print(f"✅ Short final con subtítulos dinámicos en: [green]{final_subtitled_short}[/green]")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Proceso interrumpido por el usuario.[/bold yellow]")
    except Exception as e:
        logger.exception("[bold red]¡Ha ocurrido un error fatal![/bold red]")
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    main()