import os
from dotenv import load_dotenv

load_dotenv()


# --- PDF ---
PDF_PATH = os.getenv("PDF_PATH", "me.pdf")

# --- Whisper (STT) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Edge TTS ---
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")

# --- RAG ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4))

# --- Audio files directory ---
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)