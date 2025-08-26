import openai
from dotenv import load_dotenv
import os
import json
import logging
from typing import Dict, Any, Tuple

load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API")
if not openai.api_key:
    raise ValueError("OPENAI_API not found in .env file.")

logger = logging.getLogger("rich")

def _call_openai_api(system_prompt: str, user_prompt: str, model: str = "gpt-4o-2024-05-13") -> Dict[str, Any] | None:
    """
    Helper function to call the OpenAI ChatCompletion API and parse the JSON response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from OpenAI response: {e}\nResponse content: {content}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in _call_openai_api: {e}")
    return None

def get_highlight(transcription: str, min_duration: float = 50, max_duration: float = 70, max_retries: int = 3) -> Tuple[float, float]:
    """
    Identifies the most viral highlight from a transcription using an LLM,
    ensuring the duration is within a specified range.
    Returns the start and end times in seconds.
    """
    system_prompt = f"""
    You are an expert social media video editor. The user will provide a transcription where each
    line is prefixed with its start and end times in seconds (e.g., '0.00 - 5.21: text...').
    
    Your task is to find the most interesting, engaging, or viral-worthy continuous segment.
    
    IMPORTANT CONSTRAINTS:
    1. The duration of the selected segment MUST be strictly between {min_duration} and {max_duration} seconds.
    2. To calculate the duration, subtract the start time of your first chosen line from the end time of your last chosen line.
    3. The lines you choose must be continuous (a solid block of text from the transcription).
    
    Analyze the text and its timestamps, then return a JSON object with the precise start and end times of your selected block.
    
    The JSON output must follow this exact format:
    {{
      "start_time": <float>,  // The start time of the very first line you selected.
      "end_time": <float>,    // The end time of the very last line you selected.
      "highlight_reason": "<briefly explain why you chose this segment>"
    }}
    """
    
    for attempt in range(max_retries):
        logger.info(f"Requesting highlight from LLM (Attempt {attempt + 1}/{max_retries})...")
        json_response = _call_openai_api(system_prompt, transcription)

        if json_response and isinstance(json_response, dict):
            start = json_response.get("start_time")
            end = json_response.get("end_time")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start < end:
                duration = end - start
                if min_duration <= duration <= max_duration:
                    logger.info(f"Highlight selected by LLM: {start:.2f}s to {end:.2f}s (Duration: {duration:.2f}s). Reason: {json_response.get('highlight_reason', 'N/A')}")
                    return float(start), float(end)
                else:
                    logger.warning(f"LLM returned a clip with invalid duration ({duration:.2f}s). Retrying...")
            else:
                logger.warning("LLM returned invalid start/end times. Retrying...")
        else:
            logger.warning("LLM call failed or returned invalid format. Retrying...")

    logger.error(f"Failed to get a valid highlight from the LLM after {max_retries} attempts. Returning default 0-60s.")
    # As a fallback, find the first 60-second chunk.
    # This is a simple fallback, could be improved.
    first_segment_start = 0.0
    try:
        # A bit of a hack to find the start of the first sentence in the transcription
        first_segment_start = float(transcription.strip().split(' ')[0])
    except (ValueError, IndexError):
        pass # Keep 0.0 if parsing fails
    return first_segment_start, first_segment_start + 60.0

def generate_video_metadata(highlight_text: str) -> Dict[str, Any]:
    """
    Generates a viral title, description, and hashtags for a video clip using an LLM.
    """
    system_prompt = """
    You are a social media marketing expert specializing in creating viral content for platforms
    like TikTok, YouTube Shorts, and Instagram Reels.
    
    Based on the provided text from a video clip, generate a compelling and SEO-friendly
    title, a short engaging description, and a list of relevant hashtags.
    
    The output must be a JSON object with the following structure:
    {
      "title": "<short, catchy, and viral-worthy title>",
      "description": "<a slightly longer, engaging description for the video>",
      "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"]
    }
    """
    
    logger.info("Requesting video metadata from LLM...")
    json_response = _call_openai_api(system_prompt, highlight_text)

    if json_response and isinstance(json_response, dict):
        # Basic validation
        if "title" in json_response and "description" in json_response and "hashtags" in json_response:
            logger.info("Successfully generated video metadata.")
            return json_response

    logger.error("Failed to generate valid metadata from the LLM.")
    return {
        "title": "Check out this amazing clip!",
        "description": "",
        "hashtags": ["#viral", "#clip"]
    }

if __name__ == '__main__':
    # Example usage for testing
    # Create a long dummy transcription for testing duration constraints
    dummy_transcription = ""
    for i in range(120):
        dummy_transcription += f"{i}.0 - {i+1}.0: This is sentence number {i}. \n"

    print("--- Testing GetHighlight ---")
    start_time, end_time = get_highlight(dummy_transcription)
    print(f"Highlight: {start_time:.2f}s - {end_time:.2f}s")
    print(f"Duration: {end_time - start_time:.2f}s")

    print("\n--- Testing Generate Video Metadata ---")
    metadata = generate_video_metadata("We discuss the importance of AI in modern software development.")
    print("Generated Metadata:")
    print(json.dumps(metadata, indent=2))
