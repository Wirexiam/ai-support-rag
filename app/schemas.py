from pydantic import BaseModel
from typing import List

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    context: List[str]
    latency_sec: float
