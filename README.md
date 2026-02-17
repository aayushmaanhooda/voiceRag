# Voice RAG — Talk to My Digital Twin

![Voice RAG Banner](assets/img.png)

A simple voice-based RAG (Retrieval Augmented Generation) app where you speak a question, it searches through my personal data, and responds back in audio. Like talking to a clone of me.

Built entirely with **free and open-source tools** — no paid APIs for voice (ElevenLabs alternative).

## How It Works

```
You speak → Whisper transcribes → RAG fetches context → GPT answers → Edge TTS speaks back
```

- **Speech-to-Text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — runs locally, free, fast
- **RAG**: LangChain + FAISS + OpenAI embeddings
- **Text-to-Speech**: [edge-tts](https://github.com/rany2/edge-tts) — free, natural sounding voices
- **Backend**: FastAPI
- **Frontend**: Simple HTML with real-time waveform visualizer

> Want even better voice quality or a cloned version of your own voice? You can swap edge-tts with [ElevenLabs](https://elevenlabs.io) — they offer voice cloning and premium TTS, but it's a paid service.

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/voice-rag.git
cd voice-rag
```

**2. Create virtual environment and install dependencies**
```bash
uv sync
```


**3. Add your PDF**

Drop your PDF file as `me.pdf` in the project root. This is the knowledge base the bot will answer from.

**5. Run it**
```bash
python app.py
```

**6. Open browser**
```
http://localhost:8000
```

Tap the mic, ask a question, and listen to the response.


## Tech Stack

- **faster-whisper** — 4x faster than OpenAI Whisper, runs on CPU
- **edge-tts** — Microsoft Edge's TTS, free, high quality voices
- **LangChain** — RAG pipeline
- **FAISS** — Vector similarity search
- **FastAPI** — Backend API
- **Web Audio API** — Real-time waveform visualization