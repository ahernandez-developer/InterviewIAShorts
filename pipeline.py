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
from Components.LanguageTasks import get_highlights, generate_video_metadata
from Components.FaceCropYOLO import crop_follow_face_1080x1920_yolo, mux_audio_video
from Components.Subtitles import generate_ass, burn_in_subtitles
from Components.ContentClassifier import classify_video_content
import shutil

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
            # --- 1. Setup and Download (runs once) ---
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
            wdir, main_odir = self.make_project_dirs(final_mp4)
            self.console.print(f"✅ Video descargado en: [green]{final_mp4}[/green]")
            self.logger.info(f"Directorio de trabajo: [cyan]{wdir}[/cyan]")
            self.logger.info(f"Directorio de salida principal: [cyan]{main_odir}[/cyan]")

            # --- 2. Analysis (runs once) ---
            self.hr("Análisis de Contenido")
            
            # Audio Extraction
            wav_path = wdir / "audio.wav"
            if not wav_path.exists():
                with self.console.status("[bold yellow]Extrayendo audio a WAV...[/bold yellow]", spinner="dots"):
                    extract_audio_wav(src=str(final_mp4), wav=str(wav_path))
                self.console.print("✅ Audio extraído.")
            else:
                self.console.print("✅ Audio ya extraído.")

            # Transcription
            speech_json_path = wdir / "speech.json"
            if not speech_json_path.exists():
                with self.console.status("[bold yellow]Transcribiendo audio...[/bold yellow]", spinner="dots"):
                    transcriptions = transcribeAudio(
                        str(wav_path), model_size="medium", language=None, beam_size=1,
                        vad_filter=True, diarization="auto", write_speech_json_to=str(speech_json_path),
                    )
                self.console.print("✅ Transcripción completada.")
            else:
                with open(speech_json_path, "r", encoding="utf-8") as f:
                    transcriptions = json.load(f)
                self.console.print("✅ Transcripción cargada desde archivo.")

            if not transcriptions:
                self.logger.error("[bold red]La transcripción no devolvió resultados.[/bold red]")
                return

            # Content Classification
            with self.console.status("[bold yellow]Clasificando el tipo de video...[/bold yellow]", spinner="dots"):
                trans_text_for_classification = self.build_transcript_string(transcriptions)
                video_type = classify_video_content(
                    video_path=str(final_mp4),
                    transcript_text=trans_text_for_classification,
                    video_title=title
                )
            self.console.print(f"✅ Video clasificado como: [bold green]{video_type}[/bold green]")

            # --- 3. Highlight Selection (runs once) ---
            self.hr("Selección de Highlights (LLM)")
            with self.console.status("[bold yellow]Seleccionando los mejores highlights con IA...[/bold yellow]", spinner="dots"):
                highlights = get_highlights(transcriptions)
            
            if not highlights:
                self.logger.error("[bold red]No se encontraron highlights válidos. Abortando.[/bold red]")
                return
            self.console.print(f"✅ Se encontraron [bold green]{len(highlights)}[/bold green] highlights para procesar.")

            # --- 4. Processing Loop (runs for each highlight) ---
            for i, highlight in enumerate(highlights):
                start_sec = highlight["start"]
                end_sec = highlight["end"]
                highlight_num = i + 1
                
                self.hr(f"Procesando Highlight #{highlight_num} ({start_sec:.2f}s → {end_sec:.2f}s)")
                
                # Create a dedicated output directory for this highlight
                highlight_odir = main_odir / f"highlight_{highlight_num}"
                highlight_odir.mkdir(exist_ok=True)

                # --- Metadata Generation ---
                with self.console.status(f"[bold yellow]#{highlight_num}: Generando metadatos...[/bold yellow]", spinner="dots"):
                    highlight_text = self.extract_highlight_text(transcriptions, start_sec, end_sec)
                    metadata = generate_video_metadata(highlight_text)
                
                metadata_path = highlight_odir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                self.console.print(f"✅ #{highlight_num}: Metadatos guardados.")

                # --- Video Trimming ---
                cut_path = wdir / f"cut_{highlight_num}.mp4"
                with self.console.status(f"[bold yellow]#{highlight_num}: Recortando video...[/bold yellow]", spinner="dots"):
                    trim_video_ffmpeg(
                        src=str(final_mp4), dst=str(cut_path),
                        start=float(start_sec), end=float(end_sec), copy=False
                    )
                self.console.print(f"✅ #{highlight_num}: Video recortado.")

                # --- Virtual Camera / Cropping ---
                final_short_no_subs = highlight_odir / "Final.mp4"
                if video_type in ["interview", "presentation"]:
                    cropped_path = wdir / f"cropped_{highlight_num}.mp4"
                    with self.console.status(f"[bold yellow]#{highlight_num}: Aplicando cámara virtual...[/bold yellow]", spinner="dots"):
                        crop_follow_face_1080x1920_yolo(
                            input_path=str(cut_path), output_path=str(cropped_path),
                            speech_json=str(speech_json_path), static_per_speaker=True,
                            highlight_start_sec=float(start_sec)
                        )
                    
                    with self.console.status(f"[bold yellow]#{highlight_num}: Fusionando video y audio...[/bold yellow]", spinner="dots"):
                        mux_audio_video(
                            video_with_audio=str(cut_path), video_without_audio=str(cropped_path),
                            dst=str(final_short_no_subs), fps=30
                        )
                    self.console.print(f"✅ #{highlight_num}: Cámara virtual aplicada.")
                else:
                    self.logger.info(f"#{highlight_num}: Omitiendo seguimiento de rostros para '{video_type}'.")
                    shutil.copy(str(cut_path), str(final_short_no_subs))
                    self.console.print(f"✅ #{highlight_num}: Usando recorte simple.")

                # --- Subtitle Generation ---
                final_subtitled_short = highlight_odir / "Final_subtitled.mp4"
                with self.console.status(f"[bold yellow]#{highlight_num}: Generando y quemando subtítulos...[/bold yellow]", spinner="dots"):
                    ass_path = highlight_odir / "subtitles.ass"
                    generate_ass(
                        transcriptions=transcriptions, ass_path=ass_path,
                        start_sec=start_sec, end_sec=end_sec
                    )
                    burn_in_subtitles(
                        video_path=final_short_no_subs, subtitle_path=ass_path,
                        output_path=final_subtitled_short
                    )
                self.console.print(f"✅ [bold green]Highlight #{highlight_num} completado:[/bold green] {final_subtitled_short}")

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Proceso interrumpido por el usuario.[/bold yellow]")
        except Exception as e:
            self.logger.exception("[bold red]¡Ha ocurrido un error fatal![/bold red]")

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
            # This logic can be improved for more precise text extraction based on word timestamps
            seg_start, seg_end = segment.get('start', 0), segment.get('end', 0)
            if max(start_sec, seg_start) < min(end_sec, seg_end): # Check for overlap
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
