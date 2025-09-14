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
from rich.progress import Progress
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

            with Progress(console=self.console) as progress:
                download_task = progress.add_task("[yellow]Descargando y fusionando video...[/yellow]", total=None)
                final_path_str = download_and_merge(yt, vstream, video_output_dir, progress, download_task)
                # progress.remove_task(download_task) # Remove the download task once done

                final_mp4 = Path(final_path_str)
                wdir, main_odir = self.make_project_dirs(final_mp4)

                # --- 2. Transcribe Audio (runs once) ---
                self.hr("Transcribiendo Audio")
                audio_path = wdir / "audio.wav"
                speech_json_path = wdir / "speech.json"
                
                progress.update(download_task, description="[yellow]Extrayendo audio...[/yellow]")
                extract_audio_wav(str(final_mp4), str(audio_path))
                
                transcribe_task = progress.add_task("[bold yellow]Transcribiendo audio con Whisper...[/bold yellow]", total=100)
                transcriptions = transcribeAudio(str(audio_path), progress=progress, task_id=transcribe_task, write_speech_json_to=str(speech_json_path))
                progress.update(transcribe_task, completed=100, description="✅ Audio transcrito.")
                progress.remove_task(transcribe_task)

                # --- 3. Content Classification (runs once) ---
                self.hr("Clasificación de Contenido (LLM)")
                classify_task = progress.add_task("[bold yellow]Clasificando tipo de contenido con IA...[/bold yellow]", total=100)
                video_type = classify_video_content(str(final_mp4), transcriptions, title)
                progress.update(classify_task, completed=100, description=f"✅ Contenido clasificado como: [bold green]{video_type}[/bold green]")
                progress.remove_task(classify_task)

                # --- 5. Highlight Selection (runs once) ---
            self.hr("Selección de Highlights (LLM)")
            
            highlight_selection_task = progress.add_task("[bold yellow]Seleccionando los mejores highlights con IA...[/bold yellow]", total=100)
            highlights = get_highlights(transcriptions)
            progress.update(highlight_selection_task, completed=100, description="✅ Highlights seleccionados.")
            progress.remove_task(highlight_selection_task)
            
            if not highlights:
                self.logger.error("[bold red]No se encontraron highlights válidos. Abortando.[/bold red]")
                return
            self.console.print(f"✅ Se encontraron [bold green]{len(highlights)}[/bold green] highlights para procesar.")

            # --- 6. Processing Loop (runs for each highlight) ---
            for i, highlight in enumerate(highlights):
                    start_sec = highlight["start"]
                    end_sec = highlight["end"]
                    highlight_num = i + 1
                    
                    # Create a task for the current highlight
                    highlight_task = progress.add_task(f"[bold magenta]Procesando Highlight #{highlight_num} ({start_sec:.2f}s → {end_sec:.2f}s)[/bold magenta]", total=100) # Total can be estimated or set to None

                    self.hr(f"Procesando Highlight #{highlight_num} ({start_sec:.2f}s → {end_sec:.2f}s)")
                    
                    # Create a dedicated output directory for this highlight
                    highlight_odir = main_odir / f"highlight_{highlight_num}"
                    highlight_odir.mkdir(exist_ok=True)

                    # --- Metadata Generation ---
                    progress.update(highlight_task, description=f"[bold yellow]#{highlight_num}: Generando metadatos...[/bold yellow]", completed=10)
                    highlight_text = self.extract_highlight_text(transcriptions, start_sec, end_sec)
                    metadata = generate_video_metadata(highlight_text)
                    metadata_path = highlight_odir / "metadata.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    progress.update(highlight_task, description=f"✅ #{highlight_num}: Metadatos guardados.", completed=20)

                    # --- Video Trimming ---
                    cut_path = wdir / f"cut_{highlight_num}.mp4"
                    progress.update(highlight_task, description=f"[bold yellow]#{highlight_num}: Recortando video...[/bold yellow]", completed=30)
                    trim_video_ffmpeg(
                        src=str(final_mp4), dst=str(cut_path),
                        start=float(start_sec), end=float(end_sec), copy=False
                    )
                    progress.update(highlight_task, description=f"✅ #{highlight_num}: Video recortado.", completed=40)

                    # --- Virtual Camera / Cropping ---
                    final_short_no_subs = highlight_odir / "Final.mp4"
                    if True: # video_type in ["interview", "presentation"]:
                        progress.update(highlight_task, description=f"[cyan]#{highlight_num}: Aplicando cámara virtual...[/cyan]", completed=50)
                        cropped_path = wdir / f"cropped_{highlight_num}.mp4"
                        crop_follow_face_1080x1920_yolo(
                            input_path=str(cut_path), output_path=str(cropped_path),
                            speech_json=str(speech_json_path), static_per_speaker=True,
                            highlight_start_sec=float(start_sec),
                            progress=progress, task_id=highlight_task
                        )
                        progress.update(highlight_task, description=f"[cyan]#{highlight_num}: Fusionando video y audio...[/cyan]", completed=70)
                        mux_audio_video(
                            video_with_audio=str(cut_path), video_without_audio=str(cropped_path),
                            dst=str(final_short_no_subs), fps=30
                        )
                    else:
                        self.logger.info(f"#{highlight_num}: Omitiendo seguimiento de rostros para '{video_type}'.")
                        shutil.copy(str(cut_path), str(final_short_no_subs))
                        self.console.print(f"✅ #{highlight_num}: Usando recorte simple.")
                    progress.update(highlight_task, completed=80) # After cropping/muxing

                    # --- Subtitle Generation ---
                    final_subtitled_short = highlight_odir / "Final_subtitled.mp4"
                    progress.update(highlight_task, description=f"[bold yellow]#{highlight_num}: Generando y quemando subtítulos...[/bold yellow]", completed=90)
                    ass_path = highlight_odir / "subtitles.ass"
                    generate_ass(
                        transcriptions=transcriptions,
                        ass_path=ass_path,
                        start_sec=start_sec,
                        end_sec=end_sec
                    )
                    burn_in_subtitles(
                        video_path=final_short_no_subs, subtitle_path=ass_path,
                        output_path=final_subtitled_short
                    )
                    progress.update(highlight_task, description=f"✅ [bold green]Highlight #{highlight_num} completado:[/bold green] {final_subtitled_short}", completed=100)
                    progress.remove_task(highlight_task) # Remove task once highlight is done

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
