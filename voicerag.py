"""
Voice RAG - Aayushmaan's Digital Twin
======================================
Run: python voice_rag.py

Flow: You speak → Whisper transcribes → RAG answers → Edge TTS speaks back

Requirements:
    pip install faster-whisper edge-tts sounddevice soundfile
    pip install langchain langchain-openai langchain-community faiss-cpu
    pip install python-dotenv pypdf
"""

import asyncio
import os
import sounddevice as sd
import soundfile as sf
import edge_tts
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# ============================================
# 1. RAG SETUP
# ============================================

def build_rag_chain(pdf_path: str = "me.pdf"):
    """Load PDF, create vector store, and return the RAG chain."""
    print("Loading PDF and building RAG chain...")

    # Load PDF
    docs = PyPDFLoader(pdf_path).load()

    # Split into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    ).split_documents(docs)

    # Clean encoding issues
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Create vector store + retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # LLM
    llm = init_chat_model("gpt-4o")

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a virtual version of Aayushmaan — his digital twin that talks exactly like him. 
You are chatting with a friend who is asking about Aayushmaan's life, work, skills, and experiences.

PERSONALITY:
- Talk like a real person in a casual conversation, not like a formal AI assistant
- Be witty, throw in light humor naturally — like a friend would over coffee
- Keep responses SHORT and conversational — 2-3 sentences max unless the question needs more detail
- Be confident but not arrogant about achievements
- Use casual language: "yeah", "nah", "honestly", "haha", "mate" etc.
- Show genuine enthusiasm when talking about things Aayushmaan is passionate about (AI, F1, tech)

STRICT RULES:
- NEVER use emojis — this response will be converted to speech audio
- NEVER use bullet points, numbered lists, or markdown formatting
- NEVER use special characters like asterisks, hashtags, or dashes for formatting
- Write in plain conversational sentences only
- If the context doesn't have the answer or you're unsure, say something like "mm I think I am not sure about this one, you should directly call Aayushmaan or talk to him for this"
- NEVER make up answers — if it's not in the context, just be upfront about not knowing
- NEVER break character — you ARE Aayushmaan, use "I" and "my" not "he" or "his"
- When the user says bye, goodbye, see you, or seems to be ending the conversation, respond warmly with something like "It was nice talking to you mate, all the best!" and keep it short and friendly

CONTEXT FROM AAYUSHMAAN'S DATA:
{context}

Friend's Question: {question}"""
    )

    # Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready!")
    return chain


# ============================================
# 2. SPEECH-TO-TEXT (Record + Transcribe)
# ============================================

print("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded!")


def record_and_transcribe(duration: int = 10, sample_rate: int = 16000) -> str:
    """Record audio from mic and transcribe using faster-whisper."""
    print(f"\nRecording for {duration} seconds... Speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording done!")

    sf.write("user_audio.wav", audio, sample_rate)

    segments, _ = whisper_model.transcribe("user_audio.wav", beam_size=5, vad_filter=True)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"You said: {text}")
    return text


# ============================================
# 3. TEXT-TO-SPEECH
# ============================================

async def speak(text: str, output_file: str = "response.mp3"):
    """Convert text to speech and play it."""
    communicate = edge_tts.Communicate(text, "en-US-AndrewMultilingualNeural")
    await communicate.save(output_file)

    # Play the audio
    data, sample_rate = sf.read(output_file)
    sd.play(data, sample_rate)
    sd.wait()


# ============================================
# 4. MAIN LOOP
# ============================================

async def main():
    # Build RAG chain once
    chain = build_rag_chain("me.pdf")

    print("\n" + "=" * 50)
    print("  Aayushmaan's Voice RAG - Digital Twin")
    print("=" * 50)
    print("Press Enter to start talking, type 'q' to quit.\n")

    while True:
        user_input = input(">> Press Enter to speak (or 'q' to quit): ").strip()
        if user_input.lower() == "q":
            print("See ya mate!")
            break

        # Step 1: Record and transcribe
        question = record_and_transcribe(duration=10)

        if not question:
            print("Couldn't catch that, try again.")
            continue

        # Step 2: Query RAG
        print("Thinking...")
        answer = chain.invoke(question)
        print(f"Answer: {answer}")

        # Step 3: Speak the answer
        await speak(answer)
        print()


if __name__ == "__main__":
    asyncio.run(main())