# app/main.py
from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .rag import Retriever
from .generator import Generator
from .schemas import AskRequest, AskResponse

app = FastAPI(title="AI Support RAG", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever(
    settings.index_path,
    settings.meta_path,
    settings.bm25_path,
    alpha=settings.hybrid_alpha,
    faiss_k=settings.faiss_k,
)

generator = Generator(
    url=settings.genapi_url,
    key=settings.genapi_key,
    timeout=settings.request_timeout_sec,
)


def _trim(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else (text[:limit] + "…")


def _format_fragment(doc: dict, limit: int) -> str:
    q = str(doc.get("question_ru", "")).strip()
    a = _trim(str(doc.get("answer_ru", "")).strip(), limit)
    return f"Вопрос: {q}\nОтвет: {a}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()
    try:
        docs = retriever.search(req.question, k=settings.top_k)
        context_full = [_format_fragment(d, settings.max_fragment_chars) for d in docs]
        answer = generator.ask(req.question, context_full)
        context_short = [_trim(c, settings.max_context_chars) for c in context_full]
        latency = round(time.time() - t0, 2)
        return AskResponse(answer=answer, context=context_short, latency_sec=latency)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ask_failed: {e}")
