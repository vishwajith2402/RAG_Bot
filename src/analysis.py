from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from textwrap import shorten

from .schemas import Chunk, Page


DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
CAPITALIZED_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9&.,'-]*(?:\s+[A-Z][A-Za-z0-9&.,'-]*){0,4}\b")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "shall",
    "will",
    "not",
    "you",
    "your",
    "page",
    "document",
    "section",
}


@dataclass(frozen=True)
class DocumentProfile:
    document_name: str
    pages: int
    chunks: int
    document_type: str
    summary: str
    key_topics: list[str]
    dates: list[str]
    entities: list[str]
    risks: list[str]
    clauses: list[str]
    action_items: list[str]
    pii: list[str]
    suggested_questions: list[str]


def build_document_profiles(pages: list[Page], chunks: list[Chunk]) -> list[DocumentProfile]:
    profiles = []
    document_names = sorted({page.document_name for page in pages})
    for name in document_names:
        doc_pages = [page for page in pages if page.document_name == name]
        doc_chunks = [chunk for chunk in chunks if chunk.document_name == name]
        text = " ".join(page.text for page in doc_pages)
        topics = extract_key_topics(text)
        profiles.append(
            DocumentProfile(
                document_name=name,
                pages=len(doc_pages),
                chunks=len(doc_chunks),
                document_type=infer_document_type(name, text),
                summary=summarize_text(text),
                key_topics=topics,
                dates=extract_dates(text),
                entities=extract_entities(text),
                risks=extract_risks(text),
                clauses=extract_clauses(text),
                action_items=extract_action_items(text),
                pii=detect_pii(text),
                suggested_questions=suggest_questions(name, topics, text),
            )
        )
    return profiles


def summarize_corpus(pages: list[Page], chunks: list[Chunk]) -> str:
    profiles = build_document_profiles(pages, chunks)
    if not profiles:
        return "No readable document content was found."
    lines = ["I analyzed the uploaded file set and built a searchable knowledge base.", ""]
    for profile in profiles[:5]:
        lines.append(
            f"- {profile.document_name}: {profile.document_type}, {profile.pages} page(s), "
            f"{profile.chunks} chunk(s). {profile.summary}"
        )
    if len(profiles) > 5:
        lines.append(f"- {len(profiles) - 5} additional file(s) are indexed.")
    lines.extend(["", "You can ask for summaries, risks, obligations, dates, comparisons, or exact references."])
    return "\n".join(lines)


def extract_key_topics(text: str, limit: int = 8) -> list[str]:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    counts = Counter(word for word in words if word not in STOP_WORDS and len(word) > 3)
    return [word for word, _ in counts.most_common(limit)]


def extract_dates(text: str, limit: int = 8) -> list[str]:
    seen = []
    for match in DATE_PATTERN.findall(text):
        if match not in seen:
            seen.append(match)
        if len(seen) >= limit:
            break
    return seen


def extract_entities(text: str, limit: int = 8) -> list[str]:
    blocked = {"The", "This", "That", "Section", "Page", "Table", "Figure"}
    counts = Counter(
        entity.strip(" .,")
        for entity in CAPITALIZED_PATTERN.findall(text)
        if entity.strip(" .,") not in blocked and len(entity.strip(" .,")) > 2
    )
    return [entity for entity, _ in counts.most_common(limit)]


def infer_document_type(file_name: str, text: str) -> str:
    lowered = f"{file_name} {text[:4000]}".lower()
    if any(term in lowered for term in ("agreement", "contract", "party", "termination", "liability")):
        return "contract or legal document"
    if any(term in lowered for term in ("policy", "procedure", "employee", "compliance")):
        return "policy or procedure"
    if any(term in lowered for term in ("abstract", "methodology", "references", "study", "results")):
        return "research or technical paper"
    if any(term in lowered for term in ("invoice", "amount", "payment", "total", "tax")):
        return "financial or billing document"
    if any(term in lowered for term in ("def ", "class ", "import ", "function", "const ")):
        return "source code"
    if file_name.lower().endswith((".csv", ".tsv", ".json", ".jsonl")):
        return "structured data file"
    return "general text document"


def summarize_text(text: str, width: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable preview text was available."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    candidate = " ".join(sentences[:3]).strip()
    return shorten(candidate or cleaned, width=width, placeholder="...")


def suggest_questions(file_name: str, topics: list[str], text: str) -> list[str]:
    questions = [
        "Summarize this document in simple terms",
        "What are the most important points?",
        "List all important dates and deadlines",
        "What risks, obligations, or action items are mentioned?",
    ]
    lowered = text[:5000].lower()
    if any(term in lowered for term in ("agreement", "contract", "termination", "liability")):
        questions.insert(1, "What are the key contract clauses?")
        questions.insert(2, "What termination terms are mentioned?")
    if len(topics) >= 2:
        questions.append(f"What does the document say about {topics[0]} and {topics[1]}?")
    questions.append(f"Give me exact references from {file_name}")
    return questions[:6]


def extract_risks(text: str, limit: int = 8) -> list[str]:
    risk_terms = (
        "liability",
        "penalty",
        "breach",
        "default",
        "indemn",
        "terminate",
        "confidential",
        "non-compliance",
        "warranty",
        "damages",
        "risk",
    )
    return _sentences_matching(text, risk_terms, limit)


def extract_clauses(text: str, limit: int = 10) -> list[str]:
    clause_terms = (
        "termination",
        "payment",
        "confidentiality",
        "liability",
        "indemnification",
        "governing law",
        "jurisdiction",
        "renewal",
        "warranty",
        "assignment",
        "notice",
    )
    found = []
    lowered = text.lower()
    for term in clause_terms:
        if term in lowered:
            found.append(term.title())
        if len(found) >= limit:
            break
    return found


def extract_action_items(text: str, limit: int = 8) -> list[str]:
    action_terms = ("must", "shall", "required to", "responsible for", "will provide", "deadline", "due")
    return _sentences_matching(text, action_terms, limit)


def detect_pii(text: str, limit: int = 10) -> list[str]:
    patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "ssn-like": r"\b\d{3}-\d{2}-\d{4}\b",
        "account-like": r"\b(?:account|acct|iban|routing)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
    }
    findings = []
    for label, pattern in patterns.items():
        for _ in re.finditer(pattern, text, re.IGNORECASE):
            if label not in findings:
                findings.append(label)
            if len(findings) >= limit:
                return findings
    return findings


def build_timeline(pages: list[Page], limit: int = 40) -> list[dict[str, str | int]]:
    events = []
    for page in pages:
        sentences = re.split(r"(?<=[.!?])\s+", page.text)
        for sentence in sentences:
            for date in DATE_PATTERN.findall(sentence):
                events.append(
                    {
                        "date": date,
                        "document": page.document_name,
                        "page": page.page_number,
                        "event": shorten(re.sub(r"\s+", " ", sentence), width=220, placeholder="..."),
                    }
                )
                if len(events) >= limit:
                    return events
    return events


def build_knowledge_graph(profiles: list[DocumentProfile]) -> list[dict[str, str]]:
    edges = []
    for profile in profiles:
        for entity in profile.entities[:5]:
            edges.append({"source": profile.document_name, "relation": "mentions", "target": entity})
        for clause in profile.clauses[:5]:
            edges.append({"source": profile.document_name, "relation": "contains clause", "target": clause})
        for risk in profile.risks[:3]:
            edges.append({"source": profile.document_name, "relation": "risk evidence", "target": risk})
    return edges[:60]


def _sentences_matching(text: str, terms: tuple[str, ...], limit: int) -> list[str]:
    matches = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        lowered = sentence.lower()
        if any(term in lowered for term in terms):
            excerpt = shorten(sentence.strip(), width=220, placeholder="...")
            if excerpt and excerpt not in seen:
                matches.append(excerpt)
                seen.add(excerpt)
        if len(matches) >= limit:
            break
    return matches
