# config.py
import os
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL")
REASONING_MODEL = os.environ.get("DEEPSEEK_REASONING_MODEL")
URL   = os.environ.get("DEEPSEEK_URL")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")
