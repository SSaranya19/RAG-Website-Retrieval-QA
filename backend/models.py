from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Citation(BaseModel):
    text: str
    url: str
    score: Optional[float] = None
    chunk_id: Optional[str] = None

class Answer(BaseModel):
    content: str

class ChatResponse(BaseModel):
    answer: Answer
    citations: List[Citation] = []

class IndexRequest(BaseModel):
    urls: List[str]
    mode: Literal["append", "replace"] = "replace"   # enforce valid modes

class FeedbackRequest(BaseModel):
    user_input: str
    bot_response: str
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None
    timestamp: Optional[datetime] = None
