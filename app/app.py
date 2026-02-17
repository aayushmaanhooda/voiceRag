import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


import config
from routes import router
from rag import build_rag_chain
from stt import load_whisper_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup so first request is instant."""
    load_whisper_model()
    build_rag_chain()
    print("\nServer ready! All models loaded and cached.")
    yield
    # Cleanup audio files on shutdown
    for f in os.listdir(config.AUDIO_DIR):
        os.remove(os.path.join(config.AUDIO_DIR, f))


app = FastAPI(
    title="Voice RAG API",
    description="Aayushmaan's Digital Twin - Ask questions via text or voice",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
