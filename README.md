# Ask My PDF Bot

An offline-friendly Retrieval-Augmented Generation chatbot for asking questions across PDFs and Word documents. It extracts document text page by page, chunks it, embeds it locally, retrieves relevant evidence with hybrid semantic + keyword search, and answers with source citations.

## Features

- Upload PDFs, Word documents, text files, Markdown, CSV, JSON, HTML, XML, logs, config files, and source code
- Optional image OCR for screenshots and scanned images when Tesseract is installed
- Excel, PowerPoint, email, structured JSON, and optional audio/video transcription support
- Scanned PDF OCR fallback when PyMuPDF, Pillow, pytesseract, and Tesseract are available
- Auto-analyze uploaded files and start a chat with an initial document summary
- Chat-first interface with uploads, settings, collections, and diagnostics in the sidebar
- Compact file pills, status indicators, source drawer, and document preview panel
- Suggested questions after analysis
- Prompt library for summaries, legal review, evidence lookup, and comparisons
- Ask across all files or selected documents only
- Answer style selector and source strictness selector
- Persona selector for legal, compliance, research, tutoring, and executive briefing workflows
- Optional reranking for stronger evidence ordering
- Semantic search page for inspecting passages without answer generation
- Document intelligence panels for risks, clauses, action items, timelines, graph relationships, and PII indicators
- Page-aware citations for every answer
- Hybrid retrieval: FAISS semantic search plus BM25 keyword search
- Local embeddings with `BAAI/bge-small-en-v1.5` by default
- Optional local LLM through Ollama
- Optional Gemini support when `GOOGLE_API_KEY` is available
- Strict out-of-scope handling to reduce hallucinations
- Conversational context across turns
- CPU and RAM monitoring in the sidebar
- Saved document collections
- Persistent vector indexes for faster reloads
- User feedback logging for answer quality review
- Downloadable Markdown chat reports with sources
- Download individual answers as Markdown
- Optional FastAPI backend for API-style deployment
- Docker-ready Streamlit deployment

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, upload documents, and the app will analyze them automatically by default. You can also click **Analyze files** manually. The file picker accepts any extension. Files that contain extractable text are indexed; images require the optional OCR stack; unrelated binary files such as audio, video, archives, and executables are rejected with a clear message.

For the stronger expert stack, use Python 3.10 or 3.11 and install:

```powershell
pip install -r requirements-advanced.txt
```

For OCR, install the Tesseract OCR engine on your operating system in addition to the Python packages.

Run the optional API backend:

```powershell
pip install -r requirements-advanced.txt
uvicorn api:app --reload
```

## Optional LLM Providers

The app works without a paid API by using an extractive fallback answerer. For better natural language answers, choose one provider in the sidebar.

### Ollama

Install Ollama, pull a compact model, and keep Ollama running:

```powershell
ollama pull llama3.2:3b
```

Then select **Ollama** in the app.

### Gemini

Set an API key before launching Streamlit:

```powershell
$env:GOOGLE_API_KEY="your-key"
streamlit run app.py
```

Then select **Gemini** in the app.

## Docker

```powershell
docker build -t ask-my-pdf-bot .
docker run -p 8501:8501 ask-my-pdf-bot
```

For Gemini inside Docker, pass your key:

```powershell
docker run -p 8501:8501 -e GOOGLE_API_KEY="your-key" ask-my-pdf-bot
```

For Ollama, run Ollama on the host and set `OLLAMA_BASE_URL` if needed.

Docker Compose:

```powershell
copy .env.example .env
docker compose up --build
```

## Project Structure

```text
app.py
src/
  answer.py
  chunking.py
  config.py
  document_loader.py
  embeddings.py
  rag.py
  retrieval.py
  schemas.py
  storage.py
  system_monitor.py
sample_docs/
tests/
```

## Design Notes

The bot is intentionally strict. If the retrieval evidence is weak, it refuses to answer instead of inventing content. When an LLM is configured, the prompt instructs it to answer only from retrieved context and include citations using the supplied document and page metadata.

The default embedding model is `BAAI/bge-small-en-v1.5`, with `sentence-transformers/all-MiniLM-L6-v2` as a practical fallback option. If Sentence Transformers is unavailable, the app falls back to a local scikit-learn hashing vectorizer so the project still runs in lightweight environments. FAISS is used when available; otherwise, the app falls back to a NumPy cosine search.

## Deployment Plan

- EC2 free tier: run the Docker container on port `8501`
- S3: store uploaded source documents
- Local FAISS index: keep document vectors on the EC2 volume
- CloudWatch: collect container logs
- Optional Lambda: trigger asynchronous ingestion for large documents

## Testing

```powershell
pytest
```
