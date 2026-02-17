import os
import uuid

import edge_tts

import config


async def synthesize(text: str) -> str:
    """Convert text to speech. Returns the file path of the generated audio."""
    # Clear old audio files
    for f in os.listdir(config.AUDIO_DIR):
        os.remove(os.path.join(config.AUDIO_DIR, f))

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(config.AUDIO_DIR, filename)
    communicate = edge_tts.Communicate(text, config.TTS_VOICE)
    await communicate.save(filepath)
    return filepath


async def list_voices() -> list[dict]:
    """List available English Edge TTS voices."""
    voices = await edge_tts.list_voices()
    return [
        {"name": v["ShortName"], "gender": v["Gender"]}
        for v in voices
        if v["ShortName"].startswith("en-")
    ]