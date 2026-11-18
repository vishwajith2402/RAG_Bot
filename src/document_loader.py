from __future__ import annotations

from hashlib import sha1
from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO
from email import policy
from email.parser import BytesParser

from .config import AUDIO_VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, PRESENTATION_EXTENSIONS, SPREADSHEET_EXTENSIONS, TEXT_EXTENSIONS
from .schemas import Page


def make_document_id(name: str, content: bytes) -> str:
    digest = sha1()
    digest.update(name.encode("utf-8", errors="ignore"))
    digest.update(content)
    return digest.hexdigest()[:16]


def load_document(file_name: str, file_obj: BinaryIO | bytes) -> list[Page]:
    content = file_obj if isinstance(file_obj, bytes) else file_obj.read()
    suffix = Path(file_name).suffix.lower()
    document_id = make_document_id(file_name, content)

    if suffix == ".pdf":
        return _load_pdf(file_name, document_id, content)
    if suffix == ".docx":
        return _load_docx(file_name, document_id, content)
    if suffix in SPREADSHEET_EXTENSIONS:
        return _load_spreadsheet(file_name, document_id, content)
    if suffix in PRESENTATION_EXTENSIONS:
        return _load_presentation(file_name, document_id, content)
    if suffix in IMAGE_EXTENSIONS:
        return _load_image_ocr(file_name, document_id, content)
    if suffix in AUDIO_VIDEO_EXTENSIONS:
        return _load_audio_video(file_name, document_id, content)
    if suffix == ".eml":
        return _load_email(file_name, document_id, content)
    if suffix in TEXT_EXTENSIONS or _looks_like_text(content):
        return _load_text(file_name, document_id, content)
    raise ValueError(
        f"{file_name} does not appear to contain extractable text. "
        "PDF, DOCX, Excel, PowerPoint, email, TXT, Markdown, CSV, JSON, HTML, XML, images with OCR, "
        "audio/video with transcription, logs, config files, and source code are supported."
    )


def _load_pdf(file_name: str, document_id: str, content: bytes) -> list[Page]:
    try:
        import fitz

        pdf = fitz.open(stream=content, filetype="pdf")
        pages = []
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(document_id, file_name, index, text))
        return pages
    except Exception:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                text = _ocr_pdf_page(content, index)
            if text:
                pages.append(Page(document_id, file_name, index, text))
        return pages


def _load_docx(file_name: str, document_id: str, content: bytes) -> list[Page]:
    from docx import Document

    document = Document(BytesIO(content))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return [Page(document_id, file_name, 1, text)] if text else []


def _load_spreadsheet(file_name: str, document_id: str, content: bytes) -> list[Page]:
    try:
        from openpyxl import load_workbook
    except Exception as exc:
        raise ValueError(f"{file_name} requires openpyxl for Excel extraction.") from exc

    workbook = load_workbook(BytesIO(content), data_only=True, read_only=True)
    pages = []
    for index, sheet in enumerate(workbook.worksheets, start=1):
        rows = []
        for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            values = [str(value).strip() for value in row if value is not None and str(value).strip()]
            if values:
                rows.append(f"Row {row_number}: " + " | ".join(values))
            if len(rows) >= 1000:
                rows.append("Additional rows omitted from preview chunking.")
                break
        if rows:
            pages.append(Page(document_id, file_name, index, f"Sheet: {sheet.title}\n" + "\n".join(rows)))
    return pages


def _load_presentation(file_name: str, document_id: str, content: bytes) -> list[Page]:
    try:
        from pptx import Presentation
    except Exception as exc:
        raise ValueError(f"{file_name} requires python-pptx for PowerPoint extraction.") from exc

    deck = Presentation(BytesIO(content))
    pages = []
    for index, slide in enumerate(deck.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        notes = getattr(slide, "notes_slide", None)
        if notes:
            for shape in notes.notes_text_frame.paragraphs:
                if shape.text.strip():
                    parts.append(f"Speaker note: {shape.text.strip()}")
        if parts:
            pages.append(Page(document_id, file_name, index, "\n".join(parts)))
    return pages


def _load_image_ocr(file_name: str, document_id: str, content: bytes) -> list[Page]:
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(BytesIO(content))
        text = pytesseract.image_to_string(image).strip()
    except Exception as exc:
        raise ValueError(
            f"{file_name} is an image without currently extractable text. Install Pillow, pytesseract, "
            "and the Tesseract OCR engine to extract text from images or scanned screenshots."
        ) from exc
    return [Page(document_id, file_name, 1, text)] if text else []


def _load_audio_video(file_name: str, document_id: str, content: bytes) -> list[Page]:
    try:
        import tempfile
        import whisper

        suffix = Path(file_name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
            temp.write(content)
            temp.flush()
            model = whisper.load_model("base")
            result = model.transcribe(temp.name)
        text = (result.get("text") or "").strip()
    except Exception as exc:
        raise ValueError(
            f"{file_name} requires optional Whisper transcription support for audio/video files."
        ) from exc
    return [Page(document_id, file_name, 1, text)] if text else []


def _load_email(file_name: str, document_id: str, content: bytes) -> list[Page]:
    message = BytesParser(policy=policy.default).parsebytes(content)
    headers = [
        f"From: {message.get('from', '')}",
        f"To: {message.get('to', '')}",
        f"Subject: {message.get('subject', '')}",
        f"Date: {message.get('date', '')}",
    ]
    bodies = []
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                bodies.append(part.get_content())
    elif message.get_content_type() == "text/plain":
        bodies.append(message.get_content())
    text = "\n".join(headers + bodies).strip()
    return [Page(document_id, file_name, 1, text)] if text else []


def _load_text(file_name: str, document_id: str, content: bytes) -> list[Page]:
    text = _decode_text(content)
    suffix = Path(file_name).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        text = _normalize_json_text(text, json_lines=suffix == ".jsonl")
    if suffix in {".html", ".htm", ".xml"}:
        text = _strip_markup(text)
    if suffix == ".rtf":
        text = _strip_rtf(text)
    text = text.strip()
    return [Page(document_id, file_name, 1, text)] if text else []


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _looks_like_text(content: bytes) -> bool:
    if not content:
        return False
    sample = content[:4096]
    if b"\x00" in sample:
        return False
    text_bytes = sum(byte in b"\n\r\t\b\f" or 32 <= byte <= 126 for byte in sample)
    return text_bytes / len(sample) > 0.85


def _strip_markup(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def _strip_rtf(text: str) -> str:
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\d* ?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text)


def _normalize_json_text(text: str, json_lines: bool = False) -> str:
    try:
        import json

        if json_lines:
            objects = [json.loads(line) for line in text.splitlines() if line.strip()]
            return "\n".join(_flatten_json(obj) for obj in objects)
        return _flatten_json(json.loads(text))
    except Exception:
        return text


def _flatten_json(value, prefix: str = "") -> str:
    lines = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.append(_flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            lines.append(_flatten_json(child, f"{prefix}[{index}]"))
    else:
        lines.append(f"{prefix}: {value}")
    return "\n".join(line for line in lines if line)


def _ocr_pdf_page(content: bytes, page_number: int) -> str:
    try:
        import fitz
        from PIL import Image
        import pytesseract

        pdf = fitz.open(stream=content, filetype="pdf")
        page = pdf[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""
