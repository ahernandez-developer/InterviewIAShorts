import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from Components.YoutubeDownloader import get_video_streams, download_and_merge
from Components.common_utils import create_safe_filename
from Components.Edit import extract_audio_wav, trim_video_ffmpeg
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import get_highlight, generate_video_metadata
from Components.FaceCropYOLO import crop_follow_face_1080x1920_yolo, mux_audio_video_nvenc
from Components.Subtitles import generate_ass, burn_in_subtitles

load_dotenv()

class VideoProcessingPipeline:
    def __init__(self, console: Console):
        self.console = console
        self.logger = logging.getLogger("rich")
        self.ROOT = Path(__file__).parent
        self.WORK = self.ROOT / "work"
        self.OUT = self.ROOT / "out"
        self.WORK.mkdir(exist_ok=True)
        self.OUT.mkdir(exist_ok=True)

    def run(self):
        """Executes the entire video processing pipeline."""
        try:
            self.hr("Descarga de Video")
            url = self.console.input("[bold green]Introduce la URL del video de YouTube:[/bold green] ").strip()
            
            yt, video_streams = get_video_streams(url)
            
            self.console.print("[bold]Available video streams:[/bold]")
            for idx, s in enumerate(video_streams):
                kind = "Progressive" if s.is_progressive else "Adaptive"
                size_mb = (s.filesize or 0) / (1024 * 1024)
                res = getattr(s, "resolution", None) or f"{getattr(s, 'height', '?')}p"
                self.console.print(f"{idx}. Resolution: {res}, Size: {size_mb:.2f} MB, Type: {kind}")

            choice_str = self.console.input("\nEnter the number of the video stream to download: ").strip()
            try:
                choice = int(choice_str)
                vstream = video_streams[choice]
            except (ValueError, IndexError):
                self.logger.error("[bold red]Invalid selection. Aborting.[/bold red]")
                return

            title = yt.title or "video"
            base_name = create_safe_filename(title, max_len=28)
            video_output_dir = self.ROOT / "videos" / base_name

            with self.console.status("[bold yellow]Descargando y fusionando video...[/bold yellow]", spinner="dots"):
                final_path_str = download_and_merge(yt, vstream, video_output_dir)
            
            if not final_path_str or not Path(final_path_str).exists():
                self.logger.error("[bold red]No se pudo descargar el video.[/bold red]")
                return

            final_mp4 = Path(final_path_str)
            wdir, odir = self.make_project_dirs(final_mp4)
            self.console.print(f"✅ Video descargado en: [green]{final_mp4}[/green]")
            self.logger.info(f"Directorio de trabajo: [cyan]{wdir}[/cyan]")
            self.logger.info(f"Directorio de salida: [cyan]{odir}[/cyan]")

            self.hr("Extracción de Audio")
            with self.console.status("[bold yellow]Extrayendo audio a WAV...[/bold yellow]", spinner="dots"):
                companion_audio = self.guess_companion_audio(final_mp4)
                audio_src = companion_audio if companion_audio else final_mp4
                wav_path = wdir / "audio.wav"
                extract_audio_wav(src=str(audio_src), wav=str(wav_path))
            
            if not wav_path.exists():
                self.logger.error("[bold red]No se encontró el archivo de audio para la transcripción.[/bold red]")
                return
            self.console.print(f"✅ Audio extraído a: [green]{wav_path}[/green]")

            self.hr("Transcripción (ASR) con Word Timestamps")
            speech_json_path = wdir / "speech.json"
            with self.console.status("[bold yellow]Transcribiendo audio (esto puede tardar)...[/bold yellow]", spinner="dots"):
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
                self.logger.error("[bold red]La transcripción no devolvió resultados.[/bold red]")
                return
            self.console.print(f"✅ Transcripción completada y guardada en: [green]{speech_json_path}[/green]")

            self.hr("Selección de Highlight (LLM)")
            with self.console.status("[bold yellow]Seleccionando el highlight con IA...[/bold yellow]", spinner="dots"):
                trans_text = self.build_transcript_string(transcriptions)
                start_sec, end_sec = get_highlight(trans_text)
            
            if not (isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)) and end_sec > start_sec):
                self.logger.error(f"[bold red]Ventana de highlight inválida: start={start_sec}, end={end_sec}[/bold red]")
                return
            self.console.print(f"🎯 Highlight seleccionado: [green]{start_sec:.2f}s → {end_sec:.2f}s[/green]")

            self.hr("Generación de Metadatos del Video (LLM)")
            with self.console.status("[bold yellow]Generando metadatos con IA...[/bold yellow]", spinner="dots"):
                highlight_text = self.extract_highlight_text(transcriptions, start_sec, end_sec)
                metadata = generate_video_metadata(highlight_text)
            
            metadata_path = odir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            metadata_panel = Panel(
                f"[bold]Title:[/bold] {metadata.get('title', 'N/A')}\n"
                f"[bold]Description:[/bold] {metadata.get('description', 'N/A')}\n"
                f"[bold]Hashtags:[/bold] {' '.join(metadata.get('hashtags', []))}",
                title="[bold cyan]Metadatos Generados[/bold cyan]",
                border_style="cyan",
                expand=False
            )
            self.console.print(metadata_panel)
            self.console.print(f"✅ Metadatos guardados en: [green]{metadata_path}[/green]")

            self.hr("Recorte Preciso del Video")
            cut_path = wdir / "cut.mp4"
            with self.console.status("[bold yellow]Recortando el video (re-codificando para precisión)...[/bold yellow]", spinner="dots"):
                trim_video_ffmpeg(
                    src=str(final_mp4),
                    dst=str(cut_path),
                    start=float(start_sec),
                    end=float(end_sec),
                    copy=False
                )
            self.console.print(f"✅ Video recortado en: [green]{cut_path}[/green]")

            self.hr("Cámara Virtual (Crop 9:16 + Seguimiento)")
            cropped_path = wdir / "cropped.mp4"
            with self.console.status("[bold yellow]Aplicando cámara virtual y recorte 9:16...[/bold yellow]", spinner="dots"):
                crop_follow_face_1080x1920_yolo(
                    input_path=str(cut_path),
                    output_path=str(cropped_path),
                    speech_json=str(speech_json_path),
                    static_per_speaker=True,
                    highlight_start_sec=float(start_sec) # Pass the highlight start time
                )
            self.console.print(f"✅ Clip recortado con cámara virtual: [green]{cropped_path}[/green]")

            self.hr("Muxing Final (NVENC)")
            final_short = odir / "Final.mp4"
            with self.console.status("[bold yellow]Fusionando video y audio...[/bold yellow]", spinner="dots"):
                mux_audio_video_nvenc(
                    video_with_audio=str(cut_path),
                    video_without_audio=str(cropped_path),
                    dst=str(final_short),
                    fps=30,
                    v_bitrate="6M"
                )
            self.console.print(f"✅ Short (sin subtítulos) listo en: [green]{final_short}[/green]")

            self.hr("Generación de Subtítulos Dinámicos")
            ass_path = odir / "subtitles.ass"
            final_subtitled_short = odir / f"{final_short.stem}_subtitled.mp4"
            
            with self.console.status("[bold yellow]Generando subtítulos ASS y quemándolos en el video...[/bold yellow]", spinner="dots"):
                generate_ass(
                    transcriptions=transcriptions,
                    ass_path=ass_path,
                    start_sec=start_sec,
                    end_sec=end_sec
                )
                burn_in_subtitles(
                    video_path=final_short,
                    subtitle_path=ass_path,
                    output_path=final_subtitled_short
                )
            self.console.print(f"✅ Short final con subtítulos dinámicos en: [green]{final_subtitled_short}[/green]")

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Proceso interrumpido por el usuario.[/bold yellow]")
        except Exception as e:
            self.logger.exception("[bold red]¡Ha ocurrido un error fatal![/bold red]")
            self.console.print(f"[bold red]Error: {e}[/bold red]")

    def hr(self, title: str):
        self.console.print(Panel(f"[bold blue]{title}", expand=False, border_style="blue"))

    def build_transcript_string(self, transcriptions: List[Dict[str, Any]]):
        chunks = []
        for seg in transcriptions:
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            text = seg.get('text', '')
            chunks.append(f"{start:.2f} - {end:.2f}: {text}\n")
        return "".join(chunks)

    def guess_companion_audio(self, final_mp4: Path) -> Path | None:
        folder = final_mp4.parent
        cands = sorted(list(folder.glob("audio_*.* Willow")))
        return cands[0] if cands else None

    def make_project_dirs(self, final_mp4: Path) -> tuple[Path, Path]:
        base = final_mp4.stem
        wdir = self.WORK / base
        odir = self.OUT / base
        wdir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        return wdir, odir

    def extract_highlight_text(self, transcriptions: List[Dict[str, Any]], start_sec: float, end_sec: float) -> str:
        """Extracts the text of the highlighted segment from the transcriptions."""
        highlight_text = []
        for segment in transcriptions:
            for word_info in segment.get("words", []):
                if start_sec <= word_info['start'] and word_info['end'] <= end_sec:
                    highlight_text.append(word_info['word'])
        return " ".join(highlight_text)

if __name__ == '__main__':
    # Setup for standalone testing of the pipeline
    console = Console()
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
    )
    pipeline = VideoProcessingPipeline(console)
    pipeline.run()
