from __future__ import annotations

import os
import re
from textwrap import shorten

import requests

from .config import AppConfig
from .schemas import Answer, ChatTurn, RetrievedChunk


REFUSAL = (
    "I could not find enough support for that in the uploaded documents. "
    "Try rephrasing the question or uploading a document that contains the answer."
)


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    history: list[ChatTurn],
    provider: str,
    config: AppConfig,
) -> Answer:
    confidence = retrieved[0].score if retrieved else 0.0
    if not retrieved or confidence < config.min_relevance:
        return Answer(REFUSAL, [], provider, confidence)

    provider = provider.lower()
    if provider == "gemini":
        text = _answer_with_gemini(question, retrieved, history, config)
    elif provider == "ollama":
        text = _answer_with_ollama(question, retrieved, history, config)
    else:
        text = _extractive_answer(question, retrieved)

    return Answer(text=text.strip(), sources=retrieved, used_provider=provider, confidence=confidence)


def _answer_with_gemini(
    question: str,
    retrieved: list[RetrievedChunk],
    history: list[ChatTurn],
    config: AppConfig,
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _extractive_answer(question, retrieved)
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.gemini_model)
        response = model.generate_content(_redact_pii(_build_prompt(question, retrieved, history)))
        return _verify_answer(response.text or _extractive_answer(question, retrieved), retrieved)
    except Exception as exc:
        return f"{_extractive_answer(question, retrieved)}\n\nProvider note: Gemini failed, so I used extractive mode. {exc}"


def _answer_with_ollama(
    question: str,
    retrieved: list[RetrievedChunk],
    history: list[ChatTurn],
    config: AppConfig,
) -> str:
    try:
        response = requests.post(
            f"{config.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": config.ollama_model,
                "prompt": _build_prompt(question, retrieved, history),
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=45,
        )
        response.raise_for_status()
        return _verify_answer(response.json().get("response") or _extractive_answer(question, retrieved), retrieved)
    except Exception as exc:
        return f"{_extractive_answer(question, retrieved)}\n\nProvider note: Ollama failed, so I used extractive mode. {exc}"


def _build_prompt(question: str, retrieved: list[RetrievedChunk], history: list[ChatTurn]) -> str:
    context_blocks = []
    for index, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        context_blocks.append(
            f"[Source {index}: {chunk.document_name}, page {chunk.page_number}]\n{chunk.text}"
        )
    recent_history = "\n".join(
        f"{turn.role}: {turn.content}" for turn in history[-6:] if turn.content.strip()
    )
    return f"""You are Ask My PDF Bot, a strict RAG assistant.

Rules:
- Answer only from the provided context.
- If the answer is not clearly supported, say you could not find enough support.
- Cite every factual claim with document name and page number.
- Keep the answer concise and useful.
- When possible, cite each sentence inline using the source document and page.

Conversation history:
{recent_history or "No previous turns."}

Context:
{chr(10).join(context_blocks)}

Question:
{question}

Answer:"""


def _extractive_answer(question: str, retrieved: list[RetrievedChunk]) -> str:
    lowered = question.lower()
    joined = " ".join(item.chunk.text for item in retrieved[:4])
    if any(term in lowered for term in ("summarize", "summary", "overview", "explain")):
        return _summary_answer(retrieved)
    if any(term in lowered for term in ("date", "deadline", "when")):
        date_answer = _date_answer(retrieved)
        if date_answer:
            return date_answer

    lines = [
        "Based on the uploaded documents, the strongest supported answer is in the evidence below.",
        "",
    ]
    if joined.strip():
        top = retrieved[0].chunk
        lines.append(
            f"The most relevant passage is from {top.document_name}, page {top.page_number}. "
            "I am keeping the answer grounded because no configured LLM is rewriting the evidence."
        )
        lines.append("")
    lines.append("Evidence:")
    for item in retrieved[:4]:
        chunk = item.chunk
        excerpt = shorten(chunk.text.replace("\n", " "), width=420, placeholder="...")
        lines.append(f"- {excerpt} ({chunk.document_name}, page {chunk.page_number})")
    lines.append("")
    lines.append("I am using extractive mode, so the response is limited to retrieved passages.")
    return "\n".join(lines)


def _redact_pii(text: str) -> str:
    patterns = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
        (r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[REDACTED_PHONE]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_ID]"),
        (r"\b(?:account|acct|iban|routing)\s*[:#-]?\s*[A-Z0-9-]{4,}\b", "[REDACTED_ACCOUNT]"),
    ]
    redacted = text
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
    return redacted


def _verify_answer(text: str, retrieved: list[RetrievedChunk]) -> str:
    if not text.strip():
        return text
    source_text = " ".join(item.chunk.text.lower() for item in retrieved)
    content_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{4,}", text)
        if word.lower() not in {"about", "based", "there", "their", "which", "would", "could", "should"}
    }
    if not content_words:
        return text
    supported = sum(1 for word in content_words if word in source_text)
    support_ratio = supported / max(len(content_words), 1)
    if support_ratio < 0.35:
        return (
            "I could not verify enough of the drafted answer against the retrieved sources, "
            "so I am refusing to provide it as a grounded answer.\n\n"
            + _extractive_answer("", retrieved)
        )
    return text


def _summary_answer(retrieved: list[RetrievedChunk]) -> str:
    lines = ["Here is a grounded summary from the uploaded documents:", ""]
    for item in retrieved[:5]:
        chunk = item.chunk
        excerpt = shorten(chunk.text.replace("\n", " "), width=260, placeholder="...")
        lines.append(f"- {excerpt} ({chunk.document_name}, page {chunk.page_number})")
    return "\n".join(lines)


def _date_answer(retrieved: list[RetrievedChunk]) -> str:
    pattern = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    )
    lines = ["I found these date references in the retrieved evidence:", ""]
    found = False
    seen = set()
    for item in retrieved[:6]:
        for date in pattern.findall(item.chunk.text):
            key = date.lower()
            if key in seen:
                continue
            seen.add(key)
            found = True
            lines.append(f"- {date} ({item.chunk.document_name}, page {item.chunk.page_number})")
    if not found:
        return ""
    return "\n".join(lines)
