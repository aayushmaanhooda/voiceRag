from functools import lru_cache
from faster_whisper import WhisperModel

import config


@lru_cache(maxsize=1)
def load_whisper_model() -> WhisperModel:
    """Load Whisper model once. Cached — stays in memory."""
    print("Loading Whisper model...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    print("Whisper model loaded!")
    return model


def transcribe(file_path: str) -> str:
    """Transcribe an audio file to text."""
    model = load_whisper_model()
    segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True)
    text = " ".join([seg.text for seg in segments]).strip()
    return text