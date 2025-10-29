import datetime
import asyncio
import base64
import json
import io
from time import monotonic

import pyaudio
import PIL.Image
import mss
import mss.tools
import logging
import time
import os, sys
from dotenv import load_dotenv
import multiprocessing
from logger import logger
from prompts.prompts import tools_custom, system_instruction, system_instruction_reconnection
from video_player import VideoType,  player_process
from ws_client import WebSocketClient

base_path = getattr(sys, "_MEIPASS", os.getcwd())
dotenv_path = os.path.join(base_path, ".env")

load_dotenv(dotenv_path)


from typing import List, Dict
from langchain_openai import ChatOpenAI
from schemas import FrameAnalysis, Context, DetectGameFocusPoints
from langchain_core.messages import HumanMessage

import os, ssl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

# Model and API settings
HOST = "generativelanguage.googleapis.com"
MODEL = "gemini-2.0-flash-exp"
API_KEY = os.environ.get("GEMINI_API_KEY")
URI = f"wss://{HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"

# LangChain Model Setup for Frame Analysis
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)
structured_llm_frame_analysis = model.with_structured_output(FrameAnalysis)
structured_llm_detect_game_focus_points = model.with_structured_output(DetectGameFocusPoints)
ws_client = WebSocketClient(uri=URI, logger=logger)

class GlobalContext:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.custom_state = {}
        self.game = "Unknown"
        self.game_description = ""
        self.category = "Unknown"
        self.focus_points = []
        self.notes = []
        self.frame_analysis_results = []
        self.player_goal = "" 
        self.conversation_language = None
        self.session_started = False

    def mark_session_started(self):
        self.session_started = True

    # @traceable
    def add_message(self, role: str, text: str):
        """Adds a message to the conversation history."""
        self.conversation_history.append((role, text))
        logging.info(f"[DEBUG] Message added: role={role}, text={text[:30]}...")
        logging.info(f"[DEBUG] After add: total={len(self.conversation_history)}, lang={self.conversation_language}")

    # @traceable
    def get_history(self):
        return self.conversation_history

    # @traceable
    def get_recent_conversation(self, num_pairs: int = 3) -> list:
        if not self.conversation_history:
            return []
        
        recent_messages = self.conversation_history[-(num_pairs * 2):]
        return recent_messages

    # @traceable
    def set_preference(self, key: str, value):
        self.user_preferences[key] = value

    # @traceable
    def get_preference(self, key: str, default=None):
        return self.user_preferences.get(key, default)

    # @traceable
    def to_json(self):
        return {
            "conversation_history": self.conversation_history,
            "user_preferences": self.user_preferences,
            "custom_state": self.custom_state,
            "game": self.game,
            "game_description": self.game_description,
            "category": self.category,
            "player_goal": self.player_goal,
            "conversation_language": self.conversation_language,
            "focus_points": self.focus_points,
            "notes": self.notes,
            "frame_analysis_results": self.frame_analysis_results
        }

global_context = GlobalContext()


# @traceable
def detect_game_and_focus_points(frames_data: List[str]) -> Dict:
    """Analyze random frames to detect the game and focus points."""
    logging.info("Analyzing frames to detect game and focus points...")
    message = HumanMessage(
        content=[
                    {
                        "type": "text",
                        "text": """
You are an expert gaming analyst. Analyze the following gameplay frames to detect:
1. The name of the game (if possible).
2. Key focus points to analyze during the video, such as specific tricks, objects important for the gamer community, or strategies.
Be specific, professional, and use gaming terminology.
"""
                    }
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    result = structured_llm_detect_game_focus_points.invoke([message])
    analysis = result.model_dump()
    logging.info(f"Game Detection Result: {analysis}")
    return analysis

# @traceable
def analyze_frame(frame_id: int, frames_data: List[str], context: Context) -> Dict:
    """Analyze a batch of frames and return the structured output."""
    logging.info(f"Analyzing frames {frame_id-10}-{frame_id} seconds...")
    message = HumanMessage(
        content=[
                    {
                        "type": "text",
                        "text": f"""
You are a professional game commentator analyzing a gameplay video of '{context.game}' in the category '{context.category}'.
Focus on these key points during analysis: {', '.join(context.focus_points)}.

Provide the following:
1. Frame-specific analysis:
   - Tricks, skips, glitches, and movement optimizations.
   - Mistakes or inefficiencies.
   - Execution precision and routing decisions.
2. New global notes for the game (if applicable).
"""
                    }
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in frames_data
                ]
    )

    result = structured_llm_frame_analysis.invoke([message])
    analysis_result = result.model_dump()
    analysis_result["timestamp_id"] = f"{frame_id-10}-{frame_id} seconds"
    logging.info(f"Frame Analysis Result: {analysis_result}")
    return analysis_result

# @traceable
def summarize_results(results: List[dict], context: Context) -> str:
    """Summarize the frame analysis results."""
    logging.info("Summarizing analysis results...")
    summary = {
        "total_frames_analyzed": len(results),
        "global_context": context.notes,
        "comments": [res.get("comments", "") for res in results],
        "recommendations": [res.get("recommendations", "") for res in results],
        "tricks_used": [res.get("tricks_used", "") for res in results],
        "good_actions": [res.get("good_actions", "") for res in results],
        "bad_actions": [res.get("bad_actions", "") for res in results]
    }
    summary_json = json.dumps(summary, indent=2)
    logging.info(f"Summary: {summary_json}")
    return summary_json

# @traceable
def generate_end_report(summary: str) -> str:
    """Generate an end report summarizing the entire gameplay session and providing recommendations for improvement."""
    logging.info("Generating end report...")
    report_request = f"""
Here is the structured analysis of a gameplay session:

{summary}

Write a detailed report highlighting:
- The overall gameplay and performance
- Key mistakes or inefficiencies
- Best strategies, tricks, and optimizations
- Final recommendations for improvement.
"""
    report_message = HumanMessage(content=[{"type": "text", "text": report_request}])
    result = model.invoke([report_message])
    end_report = result.content
    logging.info("End Report Generated.")
    return end_report

# -----------------------------
# Tool Functions
# -----------------------------

# @traceable
async def save_user_preferences(text):
    """Save user preferences to a file."""
    with open('user_preferences.txt', 'w') as f:
        f.write(text)
    logging.info("User preferences saved to 'user_preferences.txt'.")

# @traceable
async def remember_user_preferences(key: str, value: str):
    """Store user preferences in memory."""
    global_context.set_preference(key, value)
    logging.info(f"Preference '{key}' set to '{value}'.")

# @traceable
async def perform_game_detection(frames_data: List[str]):
    """Perform game detection and update global context."""
    analysis = await asyncio.to_thread(detect_game_and_focus_points, frames_data)
    global_context.game = analysis.get("game", "Unknown")
    global_context.focus_points = analysis.get("focus_points", [])
    logging.info(f"Detected Game: {global_context.game}")
    logging.info(f"Focus Points: {global_context.focus_points}")

# @traceable
async def update_player_goal(goal: str):
    global_context.player_goal = goal
    logging.info(f"Player goal updated: '{goal}'")

# @traceable
async def update_game_info(game_name: str = None, description: str = None):
    if game_name:
        global_context.game = game_name
        logging.info(f"Game name updated: '{game_name}'")
    if description:
        global_context.game_description = description
        logging.info(f"Game description updated: '{description}'")

# @traceable
async def update_conversation_language(language: str):
    global_context.conversation_language = language
    logging.info(f"✅ Conversation language detected and set: '{language}'")

async def handle_tool_call(ws, tool_call):
    """
    Handles tool calls from Gemini and sends corresponding responses.
    Supported tools:
    - save_user_preferences: Save user preferences
    - remember_user_preferences: Remember preferences in memory
    - perform_game_detection: Detect game from frames
    - update_player_goal: Update the current player goal
    - update_game_info: Update game information
    """
    logging.info(f"Handling tool call: {tool_call}")
    function_call = tool_call["functionCalls"][0]
    function_name = function_call["name"]
    arguments = function_call["args"]

    if function_name == "save_user_preferences":
        await save_user_preferences(str(global_context.to_json()))
        response = "User preferences saved to 'user_preferences.txt'."
    
    elif function_name == "remember_user_preferences":
        key = arguments.get("key")
        value = arguments.get("value")
        await remember_user_preferences(key, value)
        response = f"Preference '{key}' set to '{value}'."
    
    elif function_name == "perform_game_detection":
        frames = arguments.get("frames", [])
        if frames:
            await perform_game_detection(frames)
            response = f"Game detection performed. Current game: {global_context.game}."
        else:
            response = "No frames provided for game detection."
    
    elif function_name == "update_player_goal":
        goal = arguments.get("goal", "")
        if goal:
            await update_player_goal(goal)
            response = f"Player goal updated: '{goal}'"
        else:
            response = "Goal not specified."
    elif function_name == "update_game_info":
        game_name = arguments.get("game_name")
        description = arguments.get("description")
        await update_game_info(game_name=game_name, description=description)
        response = f"Game information updated. Game: '{global_context.game}'"
    
    elif function_name == "update_conversation_language":
        language = arguments.get("language", "")
        if language:
            await update_conversation_language(language)
            response = f"Language set to '{language}'. Continue speaking in this language."
        else:
            response = "Language code not provided."
    
    else:
        response = f"Function '{function_name}' is not recognized."

    msg = {
        'tool_response': {
            'function_responses': [{
                'id': function_call['id'],
                'name': function_name,
                'response': {'result': {'string_value': response}}
            }]
        }
    }

    await ws.force_send(json.dumps(msg))
    logging.info(f"Tool response sent for function '{function_name}'.")

# -----------------------------
# Real-time Assistant Class
# -----------------------------

class Agent:
    RECONNECTION_INTERVAL = 60*7  # seconds
    def __init__(self, global_context: GlobalContext, chosen_voice: str ="Fenrir"):
        self.global_context = global_context
        self.ws = None
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.collected_frames = []  # Store raw frames as base64 for analysis
        self.frame_counter = 0
        self.chosen_voice = chosen_voice
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        self.ws_client = ws_client
        self._last_connection_time: float = time.monotonic()

    async def send_to_gemini(self, message: dict):
        json_msg = json.dumps(message)
        try:
            await self.ws_client.send(json_msg)
        except (self.ws_client.WebSocketConnectionError, self.ws_client.WebSocketConnectionClosed):
            await self.ws_client.connect()
            await self.startup(tools=[{'function_declarations': tools_custom},
                                      {'google_search': {}}])
            await self.ws_client.send(json_msg)
        logging.info("Message sent to assistant.")
    
    async def receive_from_gemini(self) -> dict | None:
        try:
            message = await self.ws_client.receive()
        except (self.ws_client.WebSocketConnectionError, self.ws_client.WebSocketConnectionClosed):
            await self.ws_client.connect()
            await self.startup(tools=[{'function_declarations': tools_custom},
                                    {'google_search': {}}])
            message = await self.ws_client.receive()
        except Exception as e:
            # was: self._logger.warning(...)
            logging.warning(f"Failed to receive message: {e}")
            return None
        return json.loads(message)

    from time import monotonic

    # @traceable
    async def startup(self, tools, retry_count: int = 0):
        """
        Initialization of WebSocket connection with loading of the saved game session context.
        Upon restart restores:
        - The last 3 pairs of dialogue exchanges
        - Game information (name, description)
        - The current player goal
        - Key focus points and notes
        Args:
            tools: List of tools for Gemini
            retry_count: Retry counter (for internal use)
        """
        
        try:
            import copy
            
            has_context = (
                len(self.global_context.conversation_history) > 0
                or (self.global_context.game and self.global_context.game != "Unknown")
                or self.global_context.player_goal 
            )
            
            logging.info("="*60)
            logging.info("CONTEXT CHECK:")
            logging.info(f"  - Conversation history: {len(self.global_context.conversation_history)} messages")
            logging.info(f"  - Game: '{self.global_context.game}'")
            logging.info(f"  - Player goal: '{self.global_context.player_goal}'")
            logging.info(f"  - Language: '{self.global_context.conversation_language}'")
            logging.info(f"  - Has context: {has_context}")
            logging.info("="*60)
            
            if has_context:
                enhanced_system_instruction = copy.deepcopy(system_instruction_reconnection)
                logging.info("Using system_instruction_reconnection (reconnection mode)")
            else:
                enhanced_system_instruction = copy.deepcopy(system_instruction)
                logging.info("Using system_instruction (first start mode)")
            
            base_instruction_text = enhanced_system_instruction["parts"][0]["text"]
            
            if has_context:
                # On reconnection, take DATA from RAM (self.global_context)
                # (instructions are already in system_instruction_reconnection)
                context_parts = []
                
                if self.global_context.conversation_language:
                    lang = self.global_context.conversation_language
                    language_info = "\n\n" + "#"*80 + "\n"
                    language_info += "CRITICAL: LANGUAGE INSTRUCTION (READ FIRST!)\n"
                    language_info += "#"*80 + "\n\n"
                    language_info += f"⚠️ CONVERSATION LANGUAGE: '{lang.upper()}'\n\n"
                    language_info += f"The player speaks '{lang}' (ISO 639-1 code).\n"
                    language_info += f"Previous dialogue was in this language.\n"
                    language_info += f"YOU MUST CONTINUE speaking ONLY in '{lang}' language!\n"
                    language_info += f"DO NOT switch to English or any other language.\n"
                    language_info += f"Respond naturally in '{lang}' as if the conversation never stopped.\n\n"
                    language_info += "#"*80 + "\n"
                    context_parts.append(language_info)
                    logging.info(f"Language from RAM: {lang}")
                
                if self.global_context.game and self.global_context.game != "Unknown":
                    game_info = f"\n\n--- GAME INFORMATION (YOU ALREADY KNOW THIS!) ---\n"
                    game_info += f"Game: {self.global_context.game}\n"
                    game_info += "⚠️ DO NOT ask 'Is this [game]?' or 'Did I identify correctly?' - YOU ALREADY KNOW!\n"
                    game_info += "⚠️ DO NOT ask player to confirm the game - just use this information!\n\n"
                    
                    if self.global_context.game_description:
                        game_info += f"Description: {self.global_context.game_description}\n"
                    
                    if self.global_context.focus_points:
                        game_info += f"Key aspects: {', '.join(self.global_context.focus_points)}\n"
                    
                    game_info += "--- END OF GAME INFORMATION ---\n"
                    context_parts.append(game_info)
                    logging.info(f"Game from RAM: '{self.global_context.game}'")
                
                if self.global_context.player_goal:
                    goal_info = f"\n--- CURRENT PLAYER GOAL (YOU ALREADY KNOW THIS!) ---\n"
                    goal_info += f"Goal: {self.global_context.player_goal}\n\n"
                    goal_info += "⚠️ DO NOT ask 'What's your goal?' - YOU ALREADY KNOW!\n"
                    goal_info += "⚠️ DO NOT ask player to tell you their goal - just help achieve it!\n"
                    goal_info += "✅ IMMEDIATELY help with this goal without asking about it.\n\n"
                    goal_info += "If the player states a NEW goal, update it using the update_player_goal tool.\n"
                    goal_info += "--- END OF GOAL ---\n"
                    context_parts.append(goal_info)
                    logging.info(f"Goal from RAM: '{self.global_context.player_goal}'")
                
                recent_conv = self.global_context.get_recent_conversation(num_pairs=3)
                if recent_conv:
                    conversation_summary = "\n--- PREVIOUS DIALOGUE CONTEXT ---\n"
                    conversation_summary += "Here are the latest lines from our previous conversation:\n\n"
                    
                    for role, text in recent_conv:
                        if role == "user":
                            conversation_summary += f"Player: {text}\n"
                        elif role == "assistant":
                            conversation_summary += f"Assistant: {text}\n"
                    
                    conversation_summary += "\nContinue the dialogue naturally, taking this context into account.\n"
                    conversation_summary += "--- END OF DIALOGUE CONTEXT ---\n"
                    context_parts.append(conversation_summary)
                    logging.info(f"Conversation from RAM: {len(recent_conv)} messages")
                
                recent_notes = self.global_context.notes[-5:] if len(self.global_context.notes) > 5 else self.global_context.notes
                if recent_notes:
                    notes_info = f"\n--- IMPORTANT NOTES ---\n"
                    for i, note in enumerate(recent_notes, 1):
                        notes_info += f"{i}. {note}\n"
                    notes_info += "--- END OF NOTES ---\n"
                    context_parts.append(notes_info)
                    logging.info(f"Notes from RAM: {len(recent_notes)} notes")
                
                if context_parts:
                    enhanced_system_instruction["parts"][0]["text"] = base_instruction_text + "\n".join(context_parts)
                    logging.info(f"Context sections added: {len(context_parts)} sections")
                    logging.info(f"Total instruction length: {len(enhanced_system_instruction['parts'][0]['text'])} characters")

            setup_msg = {
                "setup": {
                    "model": f"models/{MODEL}",
                    "generation_config":
                        {
                        "speech_config":
                            {
                                "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self.chosen_voice #os.getenv("VOICE_NAME")
                            }
                        }
                            },
                        "temperature": 0,
                        },
                    "system_instruction": enhanced_system_instruction,
                    "tools": tools
                }
            }
            await self.ws_client.force_send(json.dumps(setup_msg))

            logging.info("WebSocket connection established and setup complete.")
            
        except Exception as e:
            logging.error(f"WebSocket initialization error (attempt {retry_count + 1}/3): {e}")
            
            if retry_count < 2:
                retry_delay = 2 ** retry_count
                logging.info(f"Retry attempt in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                
                try:
                    await self.ws_client.disconnect()
                    await self.ws_client.init_connect()
                except Exception as reconnect_error:
                    logging.warning(f"Reconnection error: {reconnect_error}")
                
                return await self.startup(tools, retry_count=retry_count + 1)
            else:
                logging.error("All WebSocket connection attempts exhausted")
                raise 


    # @traceable
    async def send_text(self):
        """Handle user text input and send to the model."""
        while True:  # todo: handle
            text = await asyncio.to_thread(input, "You: ")
            if text.lower() == "q":
                # When user quits, generate final report
                await self.generate_final_report()
                await self.ws_client.disconnect()
                break
            self.global_context.add_message("user", text)

            # Generate a concise summary of recent analysis
            if self.global_context.frame_analysis_results:
                latest_analysis = self.global_context.frame_analysis_results[-1]
                analysis_summary = f"Recent analysis: {json.dumps(latest_analysis, indent=2)}"
            else:
                analysis_summary = "No recent analysis available."

            # Craft the prompt with analysis summary and user message
            prompt = f"""
You are a friendly gaming assistant helping me improve my gameplay in '{self.global_context.game}'.
Based on the current situation on the screen and your analysis of my recent gameplay, provide me with strategic advice and feedback.

{analysis_summary}

User: {text}
Assistant:
"""
            msg = {
                "client_content": {
                    "turn_complete": True,
                    "turns": [{"role": "user", "parts": [{"text": prompt}]}],
                }
            }
            # await self.ws_client.force_send(json.dumps(msg))
            json_msg = json.dumps(msg)
            try:
                await self.ws_client.send(json_msg)
            except (self.ws_client.WebSocketConnectionError, self.ws_client.WebSocketConnectionClosed):
                await self.ws_client.connect()
                await self.startup(tools=[{'function_declarations': tools_custom},
                                   {'google_search': {}}])
                await self.ws_client.send(json_msg)
            logging.info("Message sent to assistant.")

    def _capture_screen_frame(self) -> str:
        """Capture a single screen frame and return it as a base64-encoded JPEG."""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        img = PIL.Image.open(io.BytesIO(image_bytes))
        img.thumbnail([1024, 1024])  # Resize for efficiency

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_encoded = base64.b64encode(image_io.read()).decode()
        return image_encoded

    # @traceable
    async def stream_screen_frames(self, interval: float = 1.0):
        """Continuously capture and send screen frames."""
        while True:
            try:
                frame = await asyncio.to_thread(self._capture_screen_frame)  # todo: handle
            except Exception as e:
                logging.warning(f"Failed to capture screen frame: {e}")
                continue
            # Collect frames for periodic analysis
            self.collected_frames.append(frame)
            self.frame_counter += 1

            # Prepare realtime input message
            msg = {"realtime_input": {"media_chunks": [
                {"mime_type": "image/jpeg", "data": frame}
            ]}}
            await self.out_queue.put(msg)

            # Periodically run background analysis every 30 frames (~30 seconds if interval=1)
            if self.frame_counter % 30 == 0:
                # Extract the last 30 frames for analysis
                frames_to_analyze = self.collected_frames[-30:]
                # Create a separate task for background analysis
                asyncio.create_task(self.run_background_analysis(frames_to_analyze))

            await asyncio.sleep(interval)

    # @traceable
    async def run_background_analysis(self, frames: List[str]):
        """
        Run background analysis on collected frames.
        Auto-detects game if unknown and saves to global_context.
        """
        logging.info("Running background analysis...")
        # Analyze last 30 frames for detailed insights
        if len(frames) >= 10:
            # Use a subset of frames for analysis to save processing
            recent_frames = frames[-10:]
            
            if self.global_context.game == "Unknown":
                try:
                    logging.info("Game not detected, attempting to detect from frames...")
                    game_detection = await asyncio.to_thread(detect_game_and_focus_points, recent_frames)
                    if game_detection.get("game") and game_detection["game"] != "Unknown":
                        self.global_context.game = game_detection["game"]
                        self.global_context.focus_points = game_detection.get("focus_points", [])
                        logging.info(f"Game auto-detected: '{self.global_context.game}'")
                        logging.info(f"Focus points: {self.global_context.focus_points}")
                except Exception as e:
                    logging.warning(f"Auto game detection error: {e}")
            
            context = Context(
                game=self.global_context.game,
                category=self.global_context.category,
                focus_points=self.global_context.focus_points,
                notes=self.global_context.notes
            )
            analysis = await asyncio.to_thread(analyze_frame, len(self.collected_frames), recent_frames, context)
            # Update global context
            if analysis.get("new_notes"):
                self.global_context.notes.append(analysis["new_notes"])
            self.global_context.frame_analysis_results.append(analysis)
            logging.info(f"Frame analysis added for frames {analysis['timestamp_id']}.")

            # **New Code to Save Analysis to Data Folder**
            # Ensure the 'data' directory exists
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/analysis_reports", exist_ok=True)

            # Define the filename with timestamp to avoid overwriting
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = f"data/analysis_reports/frame_analysis_{timestamp}.json"

            # Save the analysis_result to the JSON file
            try:
                with open(analysis_filename, 'w') as f:
                    json.dump(analysis, f, indent=2)
                logging.info(f"Frame analysis saved to '{analysis_filename}'.")
            except Exception as e:
                logging.error(f"Failed to save frame analysis: {e}")

    # @traceable
    async def listen_audio(self):
        """Capture audio from the microphone and send to the model."""
        pya_in = pyaudio.PyAudio()
        mic_info = pya_in.get_default_input_device_info()
        self.audio_stream = pya_in.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        logging.info("Audio stream started.")

        while True:
            try:  # todo: handle
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                msg = {
                    "realtime_input": {
                        "media_chunks": [
                            {
                                "data": base64.b64encode(data).decode(),
                                "mime_type": "audio/pcm",
                            }
                        ]
                    }
                }
                await self.out_queue.put(msg)
            except Exception as e:
                logging.warning(f"Failed to receive audio stream: {e}")
                continue

    # @traceable
    async def send_realtime(self):
        """Send real-time media inputs to the model."""
        while True:
            try:  # todo: handle
                msg = await self.out_queue.get()
                # await self.ws_client.force_send(json.dumps(msg))
                json_msg = json.dumps(msg)
                try:
                    await self.ws_client.send(json_msg)
                except (self.ws_client.WebSocketConnectionError, self.ws_client.WebSocketConnectionClosed):
                    continue
            except Exception as e:
                logging.warning(f"Failed to send message: {e}")
                continue

    # @traceable
    async def receive_audio(self):
        while True:
            # raw_response = await self.ws_client.force_receive()
            response_dict = await self.receive_from_gemini()
            if response_dict is None:
                continue
            # response = json.loads(raw_response)
            inline_data = (
                response_dict
                .get("serverContent", {})
                .get("modelTurn", {})
                .get("parts", [{}])[0]
                .get("inlineData", {})
                .get("data")
            )
            if inline_data:
                pcm_data = base64.b64decode(inline_data)
                self.audio_in_queue.put_nowait(pcm_data)

            text_part = (
                response_dict
                .get("serverContent", {})
                .get("modelTurn", {})
                .get("parts", [{}])[0]
                .get("text")
            )
            if text_part and text_part.strip():
                self.global_context.add_message("assistant", text_part.strip())
                logging.info(f"Assistant response added to history: {text_part[:50]}...")

            try:
                turn_complete = response_dict["serverContent"]["turnComplete"]
            except KeyError:
                continue
            else:
                if turn_complete:
                    # If you interrupt the model, it sends an end_of_turn.
                    # For interruptions to work, we need to empty out the audio queue
                    # Because it may have loaded much more audio than has played yet.
                    print("\nEnd of turn in receive audio ", time.time())
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                        print("Removed audio from queue", time.time())
                    
                    if monotonic() - self._last_connection_time > self.RECONNECTION_INTERVAL:
                        self._last_connection_time = monotonic()

                        logging.info("Reconnection time. Context remains in RAM...")
                        await self.ws_client.disconnect()
                        await self.ws_client.init_connect()
                        await self.startup(tools=[{'function_declarations': tools_custom},
                                                  {'google_search': {}}])
                        logging.info("Reconnection completed. Context preserved in memory.")

            tool_call = response_dict.get('toolCall')
            if tool_call is not None:
                await handle_tool_call(self.ws_client, tool_call)

            server_content = response_dict.get('serverContent')
            if server_content:
                self.handle_server_content(server_content)

    # @traceable
    async def play_audio(self):
        """Play received audio responses."""
        pya_out = pyaudio.PyAudio()
        stream = pya_out.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True
        )
        logging.info("Audio playback started.")

        while True:
            try:  # todo: handle
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
            except Exception as e:
                logging.warning(f"Failed to play audio: {e}")
                continue

    # @traceable
    async def periodic_context_update(self, interval: int = 60):
        """Periodically update the assistant with the global context once per minute."""
        while True:
            await asyncio.sleep(interval)
            try:  # todo: handle
                # Summarize the analysis results
                summary = await asyncio.to_thread(summarize_results, self.global_context.frame_analysis_results, Context(
                    game=self.global_context.game,
                    category=self.global_context.category,
                    focus_points=self.global_context.focus_points,
                    notes=self.global_context.notes
                ))
                # Optionally, you can send this summary to the assistant's memory or use it to influence responses
                logging.info("Periodic Context Update:")
                logging.info(summary)
                # Here, you can implement logic to update the assistant's knowledge based on the summary
            except Exception as e:
                logging.warning(f"Failed to update context: {e}")
                continue

    # @traceable
    async def generate_final_report(self):
        """Generate and save the final report summarizing the gaming session."""
        logging.info("Generating final report...")
        summary = await asyncio.to_thread(summarize_results, self.global_context.frame_analysis_results, Context(
            game=self.global_context.game,
            category=self.global_context.category,
            focus_points=self.global_context.focus_points,
            notes=self.global_context.notes
        ))
        end_report = await asyncio.to_thread(generate_end_report, summary)
        logging.info("\n--- Final Summarized Report ---")
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/summary_reports", exist_ok=True)
        with open("data/summary_reports/end_report.txt", "w") as f:
            f.write(end_report)
        logging.info("Final report saved to 'data/end_report.txt'.")

    # @traceable
    def handle_server_content(self, server_content):
        """
        Handle additional server content.
        CRITICAL: Also saves player's voice messages (userTurn) to conversation history.
        """
        logging.info(f"[DEBUG] serverContent keys: {list(server_content.keys())}")
        model_turn = server_content.get('modelTurn')
        if model_turn:
            parts = model_turn.get('parts', [])
            for part in parts:
                executable_code = part.get('executableCode')
                if executable_code:
                    logging.info("-------------------------------")
                    logging.info("```python")
                    logging.info(executable_code.get('code', ''))
                    logging.info("```")
                    logging.info("-------------------------------")

                code_execution_result = part.get('codeExecutionResult')
                if code_execution_result:
                    logging.info("-------------------------------")
                    logging.info("```")
                    logging.info(code_execution_result.get('output', ''))
                    logging.info("```")
                    logging.info("-------------------------------")
        
        user_turn = server_content.get('userTurn')
        logging.info(f"[DEBUG] userTurn present: {user_turn is not None}")

        logging.info("="*60)
        logging.info("CONTEXT CHECK:")
        logging.info(f"  - Conversation history: {len(self.global_context.conversation_history)} messages")
        logging.info(f"  - Game: '{self.global_context.game}'")
        logging.info(f"  - Player goal: '{self.global_context.player_goal}'")
        logging.info(f"  - Language: '{self.global_context.conversation_language}'")
        logging.info("="*60)
        if user_turn:
            parts = user_turn.get('parts', [])
            logging.info(f"[DEBUG] userTurn parts count: {len(parts)}")
            for part in parts:
                text_part = part.get('text')
                logging.info(f"[DEBUG] userTurn text: {text_part}")
                if text_part and text_part.strip():
                    self.global_context.add_message("user", text_part.strip())
                    logging.info(f"✅ Player voice message saved: {text_part[:50]}...")

        grounding_metadata = server_content.get('groundingMetadata')
        if grounding_metadata:
            # Handle grounding metadata if needed
            pass

    async def _start_session_once(self):
        if self.global_context.session_started:
            return

        frame_b64 = await asyncio.to_thread(self._capture_screen_frame)

        first_turn_msg = {
            "client_content": {
                "turn_complete": True,
                "turns": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": f"This is first line"
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": frame_b64
                                }
                            }
                        ]
                    }
                ]
            }
        }

        await self.ws_client.send(json.dumps(first_turn_msg))
        logging.info("START_SESSION sent")

        self.global_context.mark_session_started()

    # @traceable
    async def run_background_tasks(self, task_group: asyncio.TaskGroup):
        """Run background tasks such as periodic context updates."""
        task_group.create_task(self.periodic_context_update())

    # @traceable
    async def run(self):
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                await self.ws_client.init_connect()
                async with asyncio.TaskGroup() as tg:
                    await self.startup(tools=[{'function_declarations': tools_custom},
                                       {'google_search': {}}])
                    
                    await self._start_session_once()

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=10)

                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.stream_screen_frames())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())
                    tg.create_task(self.run_background_tasks(tg))
                
                break

            except asyncio.CancelledError:
                logging.info("Agent shutdown requested.")
                break
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Critical agent error (attempt {retry_count}/{max_retries}): {e}", exc_info=True)
                
                if retry_count < max_retries:
                    retry_delay = 2 ** retry_count
                    logging.info(f"Agent restart in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    
                    try:
                        if self.audio_stream:
                            self.audio_stream.close()
                            self.audio_stream = None
                        await self.ws_client.disconnect()
                    except Exception as cleanup_error:
                        logging.warning(f"Resource cleanup error: {cleanup_error}")
                else:
                    logging.error("All agent startup attempts exhausted")
                    if self.audio_stream:
                        self.audio_stream.close()
                    raise
                    
        try:
            await self.ws_client.disconnect()
        except Exception as disconnect_error:
            logging.warning(f"Final WebSocket disconnect error: {disconnect_error}")

def cogamer(chosen_voice="Fenrir"):
    agent = Agent(global_context=global_context, chosen_voice=chosen_voice)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logging.info("Agent terminated by user.")

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    agent = Agent(global_context=global_context, chosen_voice="Fenrir")
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logging.info("Agent terminated by user.")
