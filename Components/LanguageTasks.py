# Components/LanguageTasks.py
import os
import json
import logging
from typing import Dict, Any, List, Type
import bisect

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# --- Pydantic Schemas ---
class Highlight(BaseModel):
    start_time: float = Field(description="The start time in seconds of a potential highlight.")
    end_time: float = Field(description="The end time in seconds of a potential highlight.")
    highlight_reason: str = Field(description="A brief explanation of why this segment is engaging.")

class HighlightListResponse(BaseModel):
    highlights: List[Highlight] = Field(description="A list of the most engaging highlights found in the text.")

class MetadataResponse(BaseModel):
    title: str = Field(description="A short, catchy, and viral-worthy title for the video clip.")
    description: str = Field(description="A slightly longer, engaging description for the video.")
    hashtags: List[str] = Field(description="A list of relevant hashtags.")

# --- Gemini Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=gemini_api_key)

logger = logging.getLogger("rich")

# --- Helper Functions ---
def _call_gemini_api(prompt: str, response_schema: Type[BaseModel], model_name: str = "gemini-1.5-flash") -> Dict[str, Any] | None:
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            )
        )
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"An unexpected error occurred in _call_gemini_api: {e}")
        return None

def _adjust_highlight_duration(
    highlight: Dict[str, Any],
    all_words: List[Dict[str, Any]],
    min_duration: float,
    max_duration: float
) -> Dict[str, Any] | None:
    """
    Adjusts the start and end times of a highlight to fit the duration constraints.
    """
    if not all_words:
        return None

    word_starts = [word['start'] for word in all_words]
    start_index = bisect.bisect_left(word_starts, highlight['start_time'])
    end_index = bisect.bisect_left(word_starts, highlight['end_time'])
    
    start_index = max(0, min(start_index, len(all_words) - 1))
    end_index = max(0, min(end_index, len(all_words) - 1))

    if start_index >= end_index:
        end_index = start_index

    center_index = (start_index + end_index) // 2
    start_index = end_index = center_index

    while end_index < len(all_words) - 1 and start_index > 0:
        current_duration = all_words[end_index]['end'] - all_words[start_index]['start']
        if current_duration >= min_duration:
            break
        
        if start_index > 0:
            start_index -= 1
        if end_index < len(all_words) - 1:
            end_index += 1
    
    while (all_words[end_index]['end'] - all_words[start_index]['start']) > max_duration and end_index > start_index:
        end_index -= 1

    final_duration = all_words[end_index]['end'] - all_words[start_index]['start']

    if min_duration <= final_duration <= max_duration:
        logger.info(f"Adjusted highlight to {final_duration:.2f}s duration.")
        return {
            "start": all_words[start_index]['start'],
            "end": all_words[end_index]['end'],
            "reason": highlight.get('highlight_reason', 'N/A')
        }
    
    logger.warning(f"Could not adjust highlight to fit duration constraints. Final duration: {final_duration:.2f}s")
    return None

# --- Main Functions ---
def get_highlights(transcriptions: List[Dict[str, Any]], min_duration: float = 50, max_duration: float = 70, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Identifies top highlights using an LLM, then programmatically adjusts their duration.
    """
    system_prompt = """
    You are an expert social media video editor. The user will provide a transcription.
    Your task is to find up to 3 of the most interesting, engaging, or viral-worthy moments.
    Focus on the content and ignore duration. Provide a reason for each choice.
    """
    
    user_prompt_text = ""
    for seg in transcriptions:
        user_prompt_text += f"{seg['start']:.2f} - {seg['end']:.2f}: {seg['text']}\n"

    all_words = [word for seg in transcriptions for word in seg.get('words', [])]
    if not all_words:
        logger.error("Transcription contains no word-level timestamps. Cannot adjust highlights.")
        return []

    for attempt in range(max_retries):
        logger.info(f"Requesting candidate highlights from LLM (Attempt {attempt + 1}/{max_retries})...")
        
        full_prompt = f"{system_prompt}\n\n{user_prompt_text}"
        json_response = _call_gemini_api(full_prompt, response_schema=HighlightListResponse)

        if json_response and isinstance(json_response, dict) and "highlights" in json_response:
            adjusted_highlights = []
            for highlight in json_response["highlights"]:
                adjusted = _adjust_highlight_duration(highlight, all_words, min_duration, max_duration)
                if adjusted:
                    adjusted_highlights.append(adjusted)
            
            if adjusted_highlights:
                logger.info(f"Successfully processed and adjusted {len(adjusted_highlights)} highlights.")
                return adjusted_highlights
            else:
                logger.warning("LLM gave candidates, but none could be adjusted to the required duration. Retrying...")
        else:
            logger.warning("LLM call failed or returned invalid format. Retrying...")

    logger.error(f"Failed to get and adjust any valid highlights after {max_retries} attempts.")
    return []

def generate_video_metadata(highlight_text: str) -> Dict[str, Any]:
    """
    Generates a viral title, description, and hashtags for a video clip using an LLM.
    """
    system_prompt = """
    You are a social media marketing expert. Based on the provided text from a video clip,
    generate a compelling title, a short description, and relevant hashtags.
    """
    
    logger.info("Requesting video metadata from LLM...")
    json_response = _call_gemini_api(system_prompt + "\n\n" + highlight_text, response_schema=MetadataResponse)

    if json_response and isinstance(json_response, dict):
        if "title" in json_response and "description" in json_response and "hashtags" in json_response:
            logger.info("Successfully generated video metadata.")
            return json_response

    logger.error("Failed to generate valid metadata from the LLM.")
    return {
        "title": "Check out this amazing clip!",
        "description": "",
        "hashtags": ["#viral", "#clip"]
    }