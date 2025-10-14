import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 256))
    TOP_K_SKILLS = int(os.getenv("TOP_K_SKILLS", 8))
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    LLM_MODE = os.getenv("LLM_MODE", "LOCAL")  # <--- Add this line!
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
