import os
import random
import asyncio  # For async handling
import time
import google.generativeai as genai
from typing import List
from video_utils import extract_segments  # Utility function to extract video segments
from pydantic import BaseModel, Field
from schemas import Context, VideoSegmentAnalysis
from langsmith import traceable
import dotenv
import json

dotenv.load_dotenv()

# --- Configuration ---
SEGMENT_LENGTH = 20  # Process video in 20-second segments
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_CONTEXT_LENGTH = 10  # Trigger summarization after 10 segments

# Initialize Gemini Model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")


# --- Sub-Agent for 10-Second Segment Analysis ---
@traceable
async def analyze_segment(segment_path: str, context: Context, segment_id: int) -> dict:
    """Analyze a 10-second video segment and return structured results."""
    print(f"Analyzing segment {segment_id}...")

    # Upload the segment to Gemini
    print(f"Uploading video segment {segment_id}...")
    video_file = await asyncio.to_thread(genai.upload_file, path=segment_path)

    # Check if the file is ready
    while video_file.state.name == "PROCESSING":
        print('Processing video segment', end='', flush=True)
        await asyncio.sleep(10)
        video_file = await asyncio.to_thread(genai.get_file, video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Failed to process segment {segment_id}")

    # Create prompt for analysis
    prompt = f"""
    You are a professional speedrun commentator analyzing a gameplay video of '{context.game}' in the category '{context.category}'.
    Focus on these key points:
    {context.focus_points}

    Provide the following structured analysis:
      - General commentary
      - Suggestions for improvement
      - Tricks or optimizations observed
      - Actions done well
      - Mistakes or inefficiencies detected
    """

    # Make inference request
    response = await asyncio.to_thread(
        model.generate_content,
        [video_file, prompt],
        request_options={"timeout": 600},
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=VideoSegmentAnalysis,
        )
    )

    result = json.loads(response.text)
    result["segment_id"] = segment_id
    print(f"Segment {segment_id} Analysis Result: {result}")

    return result

def write_to_file(filepath: str, content: str):
    """Write content to a file."""
    with open(filepath, "w") as f:
        f.write(content)

# --- Sub-Agent for Summarization ---
@traceable
async def summarize_partial_results(results: List[dict], context: Context) -> str:
    """Summarize intermediate results to keep the context manageable."""
    print("\n--- Summarizing Partial Results ---")
    summary_prompt = f"""
    You are summarizing the following speedrun analysis data for '{context.game}' in '{context.category}'.

    Data:
    {json.dumps(results, indent=2)}

    Provide a concise summary including:
    - Key findings (tricks, good actions, bad actions)
    - General performance commentary
    - Recommendations for improvement
    """

    response = await asyncio.to_thread(model.generate_content, [summary_prompt])
    summary = response.text.strip()

    print("Partial Summary:", summary)
    context.notes.append(summary)
    return summary


# --- Main Video Analysis Function ---
@traceable
async def video_report(video_path: str):
    print("Extracting 10-second segments from video...")
    segments = await asyncio.to_thread(extract_segments, video_path, SEGMENT_LENGTH)
    print(f"Extracted {len(segments)} segments.")

    # Initialize context
    global_context = Context(
        game="Minecraft",
        category="Speedrun Analysis",
        focus_points=["Tricks", "Skips", "Glitches", "Optimizations", "Ender Dragon Fight", "Nether Travel", "Resource Gathering"],
        notes=[]
    )

    # Analyze each segment
    results = []
    for i, segment_path in enumerate(segments):
        result = await analyze_segment(segment_path, global_context, i + 1)
        results.append(result)

        # Trigger summarization when context becomes too large
        if len(results) >= MAX_CONTEXT_LENGTH:
            summary = await summarize_partial_results(results, global_context)
            print("\n--- Context Summarized ---")
            results.clear()

    # Final summarization
    print("\n--- Generating Final Summary ---")
    final_summary = await summarize_partial_results(results, global_context)
    print(final_summary)

    # Save results to file
    os.makedirs("data", exist_ok=True)  # Ensure the directory exists
    await asyncio.to_thread(write_to_file, "data/final_summary.json", final_summary)

    print("\n--- Video Analysis Completed ---")
    return final_summary


# --- Example Usage ---
if __name__ == "__main__":
    video_path = "data/videoplayback.mp4"
    asyncio.run(video_report(video_path))
