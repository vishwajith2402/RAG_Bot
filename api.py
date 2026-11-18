from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile

from src.config import AppConfig
from src.rag import RagEngine
from src.schemas import ChatTurn


app = FastAPI(title="Ask My PDF Bot API")
ENGINE: RagEngine | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: Annotated[list[UploadFile], File()]) -> dict[str, int | list[str]]:
    global ENGINE
    payload = []
    for file in files:
        payload.append((file.filename or "upload", await file.read()))
    ENGINE = RagEngine.from_uploads(payload, AppConfig())
    return {
        "documents": len(ENGINE.document_names),
        "pages": len(ENGINE.pages),
        "chunks": len(ENGINE.chunks),
        "document_names": ENGINE.document_names,
    }


@app.post("/ask")
def ask(
    question: Annotated[str, Form()],
    provider: Annotated[str, Form()] = "Extractive",
) -> dict:
    if ENGINE is None:
        return {"error": "No documents ingested yet."}
    answer = ENGINE.ask(question, history=[ChatTurn("user", question)], provider=provider)
    return {
        "answer": answer.text,
        "confidence": answer.confidence,
        "sources": [
            {
                "document": item.chunk.document_name,
                "page": item.chunk.page_number,
                "score": item.score,
                "text": item.chunk.text[:900],
            }
            for item in answer.sources
        ],
    }

