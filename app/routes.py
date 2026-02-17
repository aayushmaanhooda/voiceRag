import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

import config
from models import TextQuestion, RAGResponse
from rag import build_rag_chain
from stt import transcribe
from tts import synthesize, list_voices

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Aayushmaan's digital twin is alive!"}


@router.post("/ask", response_model=RAGResponse)
async def ask_text(body: TextQuestion):
    """Send a text question, get text answer + audio."""
    chain = build_rag_chain()
    answer = chain.invoke(body.question)
    audio_path = await synthesize(answer)
    audio_filename = os.path.basename(audio_path)

    return RAGResponse(
        question=body.question,
        answer=answer,
        audio_url=f"/audio/{audio_filename}",
    )


@router.post("/ask-voice", response_model=RAGResponse)
async def ask_voice(audio: UploadFile = File(...)):
    """Upload audio file, transcribe, query RAG, return text + audio."""

    # Save uploaded file
    temp_path = os.path.join(config.AUDIO_DIR, f"upload_{uuid.uuid4().hex}.wav")
    with open(temp_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # Transcribe
    question = transcribe(temp_path)
    os.remove(temp_path)

    if not question:
        raise HTTPException(status_code=400, detail="Could not transcribe audio. Try again.")

    # RAG
    chain = build_rag_chain()
    answer = chain.invoke(question)

    # TTS
    audio_path = await synthesize(answer)
    audio_filename = os.path.basename(audio_path)

    return RAGResponse(
        question=question,
        answer=answer,
        audio_url=f"/audio/{audio_filename}",
    )


@router.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve generated audio files."""
    filepath = os.path.join(config.AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(filepath, media_type="audio/mpeg")


@router.get("/voices")
async def get_voices():
    """List available TTS voices."""
    voices = await list_voices()
    return {"voices": voices}