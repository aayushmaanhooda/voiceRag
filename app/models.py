from pydantic import BaseModel


class TextQuestion(BaseModel):
    question: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    audio_url: str
