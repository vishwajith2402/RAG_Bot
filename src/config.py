from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    fallback_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "160"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.18"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xml",
    ".html",
    ".htm",
    ".rtf",
    ".log",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".conf",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".css",
    ".eml",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_VIDEO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}
PRESENTATION_EXTENSIONS = {".pptx"}

SUPPORTED_EXTENSIONS = (
    {".pdf", ".docx"}
    | TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | AUDIO_VIDEO_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | PRESENTATION_EXTENSIONS
)
