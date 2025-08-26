import logging
from rich.console import Console
from rich.logging import RichHandler
from pipeline import VideoProcessingPipeline

# --- Rich Console Setup ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
)

def main():
    """Main entry point for the application."""
    try:
        pipeline = VideoProcessingPipeline(console)
        pipeline.run()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()
