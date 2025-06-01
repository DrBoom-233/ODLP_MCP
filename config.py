# config.py
import os
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MODEL = os.environ.get("DEEPSEEK_MODEL")
URL   = os.environ.get("DEEPSEEK_URL")


