# Components/ContentClassifier.py
import os
import logging
import tempfile
import subprocess
import json
from pathlib import Path
from typing import List, Literal

import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("rich")

# --- Pydantic Schema for robust JSON output ---
class ClassificationResponse(BaseModel):
    """Defines the expected JSON structure for the classification response."""
    video_type: Literal['interview', 'presentation', 'general_content'] = Field(
        description="The classification of the video."
    )

def _get_video_duration(video_path: str) -> float:
    """Gets the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffprobe not found or failed; cannot get video duration for frame extraction.")
        return 0.0

def _extract_frames(video_path: str, num_frames: int, temp_dir: Path) -> List[Image.Image]:
    """Extracts frames from a video file and returns them as a list of PIL Images."""
    duration = _get_video_duration(video_path)
    if duration == 0:
        return []

    frames = []
    for i in range(num_frames):
        timestamp = (duration / (num_frames + 1)) * (i + 1)
        frame_path = temp_dir / f"frame_{i+1}.jpg"
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",  # High quality
            str(frame_path),
        ]
        try:
            # Using capture_output=True to hide ffmpeg's stdout/stderr unless an error occurs
            subprocess.run(cmd, check=True, capture_output=True)
            frames.append(Image.open(frame_path))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to extract frame at {timestamp}s: {e.stderr if isinstance(e, subprocess.CalledProcessError) else e}")
    
    return frames

def classify_video_content(video_path: str, transcript_text: str, video_title: str) -> str:
    """
    Classifies the video content into 'interview', 'presentation', or 'general_content'
    by analyzing video frames and the transcript with a multimodal LLM.

    Returns:
        The classified video type as a string.
    """
    logger.info("Classifying video content using multimodal analysis...")
    fallback_type = "interview"
    frames = []

    try:
        # Configure the model to use the Pydantic schema
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ClassificationResponse,
            )
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return fallback_type

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        try:
            # 1. Extract frames
            logger.info("Extracting representative frames from the video...")
            frames = _extract_frames(video_path, num_frames=5, temp_dir=temp_dir)
            
            if not frames:
                logger.warning("Could not extract frames. Classification will be based on text only.")

            # 2. Build the prompt
            system_prompt = """
            You are an expert video content analyst. Your task is to classify a video into one of
            the following categories based on its title, transcript, and representative frames.

            The categories are:
            - 'interview': A conversation between two or more people. Look for multiple faces, shot-reverse-shot scenes, or conversational turn-taking in the transcript.
            - 'presentation': A single person speaking to an audience (e.g., a conference talk, lecture, or monologue). Look for a single dominant speaker, slides, or a stage.
            - 'general_content': Anything else. This includes TV show clips, news reports, documentaries, vlogs with varied scenes, etc. This is the category if the content is not clearly an interview or a presentation.
            """

            # Limit transcript length to avoid overly long prompts
            transcript_summary = (transcript_text[:4000] + '...') if len(transcript_text) > 4000 else transcript_text

            prompt_parts = [
                system_prompt,
                f"Video Title: {video_title}",
                f"Video Transcript (summary):\n{transcript_summary}",
                "\n--- Video Frames ---"
            ]
            prompt_parts.extend(frames)

            # 3. Call Gemini API
            logger.info("Sending request to Gemini for classification...")
            response = model.generate_content(prompt_parts)
            
            data = json.loads(response.text)
            video_type = data.get("video_type")

            if video_type in ['interview', 'presentation', 'general_content']:
                logger.info(f"Video classified as: [bold green]{video_type}[/bold green]")
                return video_type
            else:
                logger.warning(f"Gemini returned an unknown video_type: '{video_type}'. Defaulting to '{fallback_type}'.")
                return fallback_type

        finally:
            # --- FIX: Ensure all frame file handles are closed ---
            for frame in frames:
                frame.close()
    
    # Fallback in case of temporary directory issues
    return fallback_type
