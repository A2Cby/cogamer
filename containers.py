from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from logger import logger
from ws_client import WebSocketClient
import os
HOST = "generativelanguage.googleapis.com"
API_KEY = os.environ.get("GEMINI_API_KEY")
URI = f"wss://{HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"


class CogamerContainer(DeclarativeContainer):
    logger = logger
    ws_client = providers.Singleton(WebSocketClient, uri=URI, logger=logger)  # todo: change uri
