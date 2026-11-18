from __future__ import annotations

import hashlib
import html
from pathlib import Path
import time
from textwrap import shorten

import streamlit as st

from src.analysis import build_knowledge_graph, build_timeline
from src.config import AppConfig
from src.rag import RagEngine
from src.schemas import ChatTurn
from src.storage import list_collections, load_collection, save_collection, save_uploaded_files
from src.system_monitor import get_system_stats


st.set_page_config(page_title="Ask My PDF Bot", page_icon="app", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --void: #030505;
            --panel: #08100f;
            --panel-2: #0d1816;
            --line: rgba(0, 229, 255, 0.34);
            --cyan: #00e5ff;
            --acid: #b9ff66;
            --rose: #ff4fd8;
            --text: #edf7f6;
            --muted: #9bb4b1;
        }

        .stApp {
            color: var(--text);
            background:
                linear-gradient(90deg, rgba(0, 229, 255, 0.04) 1px, transparent 1px),
                linear-gradient(rgba(185, 255, 102, 0.035) 1px, transparent 1px),
                radial-gradient(circle at 18% 10%, rgba(0, 229, 255, 0.13), transparent 28%),
                radial-gradient(circle at 82% 18%, rgba(255, 79, 216, 0.10), transparent 24%),
                #030505;
            background-size: 42px 42px, 42px 42px, auto, auto, auto;
        }

        .stAppViewContainer,
        .main,
        [data-testid="stAppViewContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            background: transparent;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        header[data-testid="stHeader"] {
            background: rgba(3, 5, 5, 0.72);
            border-bottom: 1px solid rgba(0, 229, 255, 0.16);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 16, 15, 0.98), rgba(3, 5, 5, 0.98));
            border-right: 1px solid rgba(0, 229, 255, 0.26);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text);
        }

        section[data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(13, 24, 22, 0.82);
            border: 1px solid rgba(185, 255, 102, 0.22);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        .mission-header {
            position: relative;
            padding: 0.95rem 0 0.75rem;
            border-top: 1px solid rgba(0, 229, 255, 0.38);
            border-bottom: 1px solid rgba(185, 255, 102, 0.26);
            margin-bottom: 0.85rem;
        }

        .mission-kicker {
            color: var(--acid);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }

        .mission-title {
            color: var(--text);
            font-size: clamp(1.85rem, 3.2vw, 3.3rem);
            line-height: 1;
            font-weight: 900;
            margin: 0;
            text-shadow: 0 0 18px rgba(0, 229, 255, 0.34);
        }

        .mission-copy {
            color: var(--muted);
            max-width: 760px;
            margin-top: 0.75rem;
            font-size: 1rem;
            line-height: 1.65;
        }

        .chat-shell {
            border: 1px solid rgba(0, 229, 255, 0.20);
            border-radius: 8px;
            background: rgba(3, 5, 5, 0.36);
            padding: 0.9rem;
            min-height: 640px;
        }

        .side-drawer {
            position: sticky;
            top: 5.2rem;
            border: 1px solid rgba(185, 255, 102, 0.18);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.64);
            padding: 0.85rem;
            max-height: calc(100vh - 6rem);
            overflow: auto;
        }

        .status-rail {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem;
        }

        .status-cell {
            border: 1px solid rgba(0, 229, 255, 0.24);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.78);
            padding: 0.85rem;
            min-height: 82px;
        }

        .status-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }

        .status-value {
            color: var(--acid);
            font-size: 1.15rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .file-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.5rem 0 0.7rem;
        }

        .file-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            max-width: 100%;
            border: 1px solid rgba(0, 229, 255, 0.26);
            border-radius: 6px;
            background: rgba(13, 24, 22, 0.84);
            color: var(--text);
            padding: 0.42rem 0.55rem;
            font-size: 0.83rem;
            font-weight: 700;
        }

        .file-pill span {
            color: var(--acid);
            font-size: 0.72rem;
            text-transform: uppercase;
        }

        .status-dot {
            width: 0.58rem;
            height: 0.58rem;
            border-radius: 999px;
            background: var(--acid);
            box-shadow: 0 0 12px rgba(185, 255, 102, 0.72);
            flex: 0 0 auto;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(0, 229, 255, 0.42);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.66);
            padding: 0.65rem;
        }

        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(3, 5, 5, 0.48);
            border: 1px solid rgba(0, 229, 255, 0.22);
            border-radius: 8px;
            color: var(--text);
        }

        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {
            color: var(--muted);
        }

        [data-testid="stFileUploaderFile"] {
            background: rgba(13, 24, 22, 0.92);
            border: 1px solid rgba(185, 255, 102, 0.24);
            border-radius: 8px;
            color: var(--text);
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFileUploader"] button {
            border-radius: 6px;
            border: 1px solid rgba(0, 229, 255, 0.58);
            background: linear-gradient(180deg, rgba(0, 229, 255, 0.18), rgba(185, 255, 102, 0.08));
            color: var(--text);
            font-weight: 800;
            box-shadow: 0 0 18px rgba(0, 229, 255, 0.12);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stFileUploader"] button:hover {
            border-color: var(--acid);
            color: var(--acid);
            box-shadow: 0 0 24px rgba(185, 255, 102, 0.18);
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(0, 229, 255, 0.18);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.68);
            box-shadow: inset 3px 0 0 rgba(0, 229, 255, 0.38);
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            border-color: rgba(255, 79, 216, 0.28);
            box-shadow: inset 3px 0 0 rgba(255, 79, 216, 0.58);
        }

        .stChatInput {
            border-top: 1px solid rgba(0, 229, 255, 0.18);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stChatInput textarea,
        textarea,
        input,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="input"] {
            background: rgba(3, 5, 5, 0.94);
            border-color: rgba(0, 229, 255, 0.32);
            color: var(--text);
            border-radius: 6px;
        }

        .stChatInput > div,
        [data-testid="stChatInput"] {
            background: rgba(3, 5, 5, 0.86);
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"],
        [data-testid="stSelectboxVirtualDropdown"] {
            background: #08100f;
            border: 1px solid rgba(0, 229, 255, 0.34);
            color: var(--text);
        }

        [role="option"],
        [data-baseweb="menu"] li {
            background: #08100f;
            color: var(--text);
        }

        [role="option"]:hover,
        [data-baseweb="menu"] li:hover {
            background: rgba(0, 229, 255, 0.14);
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] div {
            color: var(--text);
        }

        div[data-testid="stMarkdownContainer"] p,
        label,
        .stCaptionContainer,
        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(185, 255, 102, 0.22);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.58);
        }

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            background: rgba(8, 16, 15, 0.58);
            color: var(--text);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid rgba(0, 229, 255, 0.26);
            background: rgba(8, 16, 15, 0.86);
        }

        .analysis-panel {
            border: 1px solid rgba(0, 229, 255, 0.24);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.72);
            padding: 1rem;
            margin: 0.65rem 0 1rem;
        }

        .analysis-panel strong {
            color: var(--acid);
        }

        .empty-state {
            border: 1px solid rgba(0, 229, 255, 0.22);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(8, 16, 15, 0.80), rgba(3, 5, 5, 0.58));
            padding: 1.2rem;
            margin: 1rem 0;
        }

        .empty-title {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }

        .empty-copy {
            color: var(--muted);
            line-height: 1.6;
        }

        .timeline {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(0, 229, 255, 0.28);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.72);
            padding: 0.75rem;
            margin: 0.6rem 0;
        }

        .timeline::after {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            width: 36%;
            background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.16), transparent);
            animation: scan 1.8s linear infinite;
        }

        .timeline-steps {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.45rem;
        }

        .timeline-step {
            border: 1px solid rgba(185, 255, 102, 0.22);
            border-radius: 6px;
            padding: 0.45rem;
            color: var(--acid);
            font-size: 0.78rem;
            text-align: center;
            background: rgba(3, 5, 5, 0.48);
        }

        @keyframes scan {
            from { left: -40%; }
            to { left: 110%; }
        }

        .file-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1rem;
        }

        .file-card,
        .source-card {
            border: 1px solid rgba(0, 229, 255, 0.22);
            border-radius: 8px;
            background: rgba(8, 16, 15, 0.72);
            padding: 0.85rem;
        }

        .file-card-title,
        .source-title {
            color: var(--text);
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .file-card-meta,
        .source-meta {
            color: var(--acid);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .source-text {
            color: var(--muted);
            line-height: 1.55;
        }

        .citation-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.55rem 0 0.2rem;
        }

        .citation-badge {
            border: 1px solid rgba(185, 255, 102, 0.28);
            border-radius: 6px;
            background: rgba(185, 255, 102, 0.08);
            color: var(--acid);
            padding: 0.25rem 0.42rem;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .score-bar {
            height: 0.38rem;
            border-radius: 999px;
            background: rgba(0, 229, 255, 0.10);
            overflow: hidden;
            margin: 0.45rem 0;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--cyan), var(--acid));
        }

        code {
            color: var(--acid);
            background: rgba(185, 255, 102, 0.09);
            border: 1px solid rgba(185, 255, 102, 0.18);
            border-radius: 4px;
            padding: 0.08rem 0.25rem;
        }

        @media (max-width: 760px) {
            .status-rail {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .timeline-steps {
                grid-template-columns: 1fr;
            }

            .mission-title {
                font-size: 2.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(engine: RagEngine | None) -> None:
    pages = len(engine.pages) if engine else 0
    chunks = len(engine.chunks) if engine else 0
    model = engine.embedder.model_name if engine else "standby"
    status = "knowledge base online" if engine else "awaiting documents"
    st.markdown(
        f"""
        <section class="mission-header">
            <div class="mission-kicker">chat with uploaded files</div>
            <h1 class="mission-title">Ask My PDF Bot</h1>
            <div class="mission-copy">
                Upload files, analyze them into a private knowledge base, then ask questions in a familiar chat flow.
                Every answer is grounded in retrieved evidence from your documents.
            </div>
        </section>
        <div class="status-rail">
            <div class="status-cell">
                <div class="status-label">system</div>
                <div class="status-value">{status}</div>
            </div>
            <div class="status-cell">
                <div class="status-label">pages</div>
                <div class="status-value">{pages}</div>
            </div>
            <div class="status-cell">
                <div class="status-label">chunks</div>
                <div class="status-value">{chunks}</div>
            </div>
            <div class="status-cell">
                <div class="status-label">embedder</div>
                <div class="status-value">{model}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_theme()

    defaults = AppConfig()
    engine: RagEngine | None = st.session_state.get("engine")

    with st.sidebar:
        st.header("Files")
        uploads = st.file_uploader("Upload files", accept_multiple_files=True)
        if uploads:
            payload = [(file.name, file.getvalue()) for file in uploads]
            st.session_state["last_upload_payload"] = payload
            render_file_pills_from_payload(payload, status="uploaded")
        else:
            payload = []

        auto_analyze = st.toggle("Auto-analyze uploads", value=True)

        st.divider()
        st.header("Chat")
        provider = st.selectbox("Answer provider", ["Extractive", "Ollama", "Gemini"])
        persona = st.selectbox(
            "Persona",
            ["Document analyst", "Legal analyst", "Compliance reviewer", "Research assistant", "Student tutor", "Executive briefer"],
        )
        answer_style = st.selectbox("Answer style", ["Balanced", "Concise", "Detailed", "Bullet points", "Simple explanation"])
        strictness = st.select_slider("Source strictness", options=["Exploratory", "Balanced", "Strict"], value="Balanced")
        rerank = st.toggle("Rerank evidence", value=False)

        with st.expander("Advanced retrieval", expanded=False):
            chunk_size = st.slider("Chunk size", 500, 1600, st.session_state.get("chunk_size", defaults.chunk_size), 50)
            chunk_overlap = st.slider(
                "Chunk overlap",
                50,
                350,
                st.session_state.get("chunk_overlap", defaults.chunk_overlap),
                10,
            )
            top_k = st.slider("Retrieved chunks", 3, 12, st.session_state.get("top_k", defaults.top_k))
            min_relevance = st.slider(
                "Minimum relevance",
                0.0,
                0.6,
                st.session_state.get("min_relevance", defaults.min_relevance),
                0.01,
            )

        min_relevance = {"Exploratory": 0.08, "Strict": max(0.24, min_relevance)}.get(strictness, min_relevance)
        config = AppConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            min_relevance=min_relevance,
        )

        selected_docs: list[str] = []
        if engine:
            st.divider()
            st.subheader("Ask Scope")
            stored_docs = st.session_state.get("selected_docs", engine.document_names)
            valid_defaults = [name for name in stored_docs if name in engine.document_names] or engine.document_names
            selected_docs = st.multiselect(
                "Documents",
                engine.document_names,
                default=valid_defaults,
            )
            st.session_state["selected_docs"] = selected_docs

        st.divider()
        st.subheader("Collections")
        collections = list_collections()
        if collections:
            labels = [item["label"] for item in collections]
            selected_collection = st.selectbox("Load saved set", ["None"] + labels)
            if selected_collection != "None" and st.button("Load collection"):
                item = collections[labels.index(selected_collection)]
                collection_payload = load_collection(item["path"])
                build_engine_from_payload(collection_payload, config=config, source_label="collection")
        else:
            st.caption("No saved collections yet.")

        collection_name = st.text_input("Save current set as", value="")
        if engine and st.button("Save collection"):
            saved_payload = st.session_state.get("last_upload_payload", [])
            if saved_payload:
                save_collection(saved_payload, collection_name or None)
                st.toast("Collection saved")

        st.divider()
        if st.button("Analyze files", type="primary", disabled=not payload):
            build_engine_from_payload(payload, config, source_label="upload")
            st.session_state["last_upload_signature"] = upload_signature(payload) if payload else ""

        if st.button("Clear chat"):
            st.session_state["messages"] = []
            st.session_state["answer_records"] = []

        st.divider()
        st.subheader("System")
        stats = get_system_stats()
        if "error" in stats:
            st.caption(f"Monitoring unavailable: {stats['error']}")
        else:
            st.metric("CPU", f"{stats['cpu_percent']:.1f}%")
            st.metric("RAM", f"{stats['memory_percent']:.1f}%", f"{stats['memory_used_gb']} / {stats['memory_total_gb']} GB")

    if payload and auto_analyze:
        signature = upload_signature(payload)
        if signature != st.session_state.get("last_upload_signature"):
            build_engine_from_payload(payload, config, source_label="upload")
            st.session_state["last_upload_signature"] = signature

    engine = st.session_state.get("engine")
    render_header(engine)
    render_status_lights(engine, provider, strictness)

    engine = st.session_state.get("engine")
    if engine:
        selected_docs = st.session_state.get("selected_docs") or engine.document_names
        render_file_pills(engine, selected_docs)
        chat_tab, search_tab, insights_tab = st.tabs(["Chat", "Semantic Search", "Insights"])
        with chat_tab:
            chat_col, drawer_col = st.columns([0.68, 0.32], gap="large")
            with chat_col:
                st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
                render_prompt_library(engine)
                render_chat(provider, selected_docs, answer_style, persona, rerank)
                st.markdown("</div>", unsafe_allow_html=True)
            with drawer_col:
                render_source_drawer(engine)
        with search_tab:
            render_semantic_search(engine, selected_docs)
        with insights_tab:
            render_advanced_insights(engine)
    else:
        render_empty_state()
        render_chat(provider, None, answer_style, persona, rerank)


def build_engine(uploads, config: AppConfig) -> None:
    upload_payload = [(file.name, file.getvalue()) for file in uploads]
    build_engine_from_payload(upload_payload, config, source_label="upload")


def build_engine_from_payload(
    upload_payload: list[tuple[str, bytes]],
    config: AppConfig | None,
    source_label: str,
) -> None:
    config = config or AppConfig()
    try:
        start = time.perf_counter()
        render_analysis_timeline()
        progress = st.progress(0, text="Reading files")
        with st.spinner("Analyzing files and preparing the chat context..."):
            progress.progress(20, text="Extracting text")
            index_path = Path("data/indexes") / f"{upload_signature(upload_payload)}.pkl"
            if index_path.exists():
                progress.progress(55, text="Loading saved vector index")
                engine = RagEngine.load_index(index_path, config)
            else:
                engine = RagEngine.from_uploads(upload_payload, config)
            progress.progress(78, text="Building retrieval index")
            elapsed = time.perf_counter() - start
            target = Path("data/uploads") if source_label == "upload" else Path("data/loaded_collection")
            save_uploaded_files(upload_payload, target)
            engine.save_index(index_path)
            progress.progress(100, text="Ready to chat")
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["engine"] = engine
    st.session_state["messages"] = [{"role": "assistant", "content": build_analysis_message(engine)}]
    st.session_state["answer_records"] = []
    st.session_state["last_upload_payload"] = upload_payload
    st.toast(f"Knowledge base built in {elapsed:.1f}s")


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def build_analysis_message(engine: RagEngine) -> str:
    lines = ["I analyzed your uploads. Here is what I found:", ""]
    for profile in engine.profiles[:5]:
        topics = ", ".join(profile.key_topics[:5]) or "general content"
        lines.append(
            f"- {profile.document_name}: {profile.document_type}; {profile.pages} page(s). "
            f"Topics: {topics}. {profile.summary}"
        )
    if len(engine.profiles) > 5:
        lines.append(f"- {len(engine.profiles) - 5} more file(s) are indexed.")
    lines.extend(["", "Ask me a question and I will answer from the uploaded content with sources."])
    return "\n".join(lines)


def render_status_lights(engine: RagEngine | None, provider: str, strictness: str) -> None:
    knowledge = "ready" if engine else "waiting"
    llm = provider.lower()
    st.markdown(
        f"""
        <div class="file-pill-row">
            <div class="file-pill"><div class="status-dot"></div>Knowledge <span>{knowledge}</span></div>
            <div class="file-pill"><div class="status-dot"></div>Provider <span>{html.escape(llm)}</span></div>
            <div class="file-pill"><div class="status-dot"></div>Sources <span>{html.escape(strictness)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_pills_from_payload(payload: list[tuple[str, bytes]], status: str) -> None:
    pills = []
    for name, content in payload[:8]:
        safe_name = html.escape(shorten(name, width=30, placeholder="..."))
        size = format_bytes(len(content))
        pills.append(
            f"""
            <div class="file-pill">
                <div class="status-dot"></div>{safe_name}<span>{html.escape(status)} | {size}</span>
            </div>
            """
        )
    if len(payload) > 8:
        pills.append(f"<div class='file-pill'>+{len(payload) - 8} more<span>queued</span></div>")
    st.markdown(f"<div class='file-pill-row'>{''.join(pills)}</div>", unsafe_allow_html=True)


def render_file_pills(engine: RagEngine, selected_docs: list[str]) -> None:
    selected = set(selected_docs)
    pills = []
    for profile in engine.profiles:
        status = "active" if profile.document_name in selected else "filtered"
        name = html.escape(shorten(profile.document_name, width=34, placeholder="..."))
        pills.append(
            f"""
            <div class="file-pill">
                <div class="status-dot"></div>{name}<span>{status} | {profile.pages}p</span>
            </div>
            """
        )
    st.markdown(f"<div class='file-pill-row'>{''.join(pills)}</div>", unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">Upload files to begin.</div>
            <div class="empty-copy">
                Drop PDFs, Word documents, notes, code, CSV, JSON, or text files into the sidebar.
                I will analyze them, create a private retrieval index, and answer questions with sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Try these once files are analyzed")
    cols = st.columns(3)
    starters = [
        "Summarize the uploaded files",
        "Find obligations and risks",
        "Extract important dates",
        "Compare the documents",
        "Explain this simply",
        "Show the strongest sources",
    ]
    for index, starter in enumerate(starters):
        with cols[index % 3]:
            if st.button(starter, key=f"empty-starter-{index}"):
                st.session_state["pending_question"] = starter


def render_analysis_timeline() -> None:
    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-steps">
                <div class="timeline-step">Reading</div>
                <div class="timeline-step">Extracting</div>
                <div class="timeline-step">Profiling</div>
                <div class="timeline-step">Embedding</div>
                <div class="timeline-step">Ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_library(engine: RagEngine) -> None:
    with st.expander("Prompt library", expanded=False):
        categories = {
            "Summary": [
                "Give me a concise executive summary",
                "Explain the uploaded files in simple terms",
                "What are the most important points?",
            ],
            "Legal Review": [
                "What obligations are mentioned?",
                "What are the key risks?",
                "What termination or liability terms appear?",
            ],
            "Evidence": [
                "List the strongest evidence with page references",
                "Find all important dates and deadlines",
                "Which pages should I read first?",
            ],
            "Compare": [
                "Compare the uploaded documents",
                "What topics appear across multiple files?",
                "What is different between the documents?",
            ],
        }
        tabs = st.tabs(list(categories))
        for tab, (_, prompts) in zip(tabs, categories.items()):
            with tab:
                for index, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"library-{prompt}-{index}"):
                        st.session_state["pending_question"] = prompt
        if engine.profiles:
            st.caption("Document-specific")
            for index, question in enumerate(engine.profiles[0].suggested_questions[:3]):
                if st.button(question, key=f"profile-prompt-{index}"):
                    st.session_state["pending_question"] = question


def render_source_drawer(engine: RagEngine) -> None:
    st.markdown("<div class='side-drawer'>", unsafe_allow_html=True)
    st.subheader("Source Drawer")
    records = st.session_state.get("answer_records", [])
    if records and records[-1]["sources"]:
        st.caption("Latest retrieved evidence")
        for source in records[-1]["sources"][:5]:
            score = max(0.0, min(1.0, float(source["score"])))
            width = int(score * 100)
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{html.escape(source["document"])}</div>
                    <div class="source-meta">Page {source["page"]} | score {score:.3f}</div>
                    <div class="score-bar"><div class="score-fill" style="width:{width}%"></div></div>
                    <div class="source-text">{html.escape(shorten(source["text"].replace(chr(10), " "), width=280, placeholder="..."))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("Ask a question to populate retrieved source excerpts.")

    st.divider()
    st.subheader("Preview")
    names = engine.document_names
    selected_name = st.selectbox("Document", names, key="preview-doc")
    pages = [page for page in engine.pages if page.document_name == selected_name]
    page_numbers = [page.page_number for page in pages]
    selected_page = st.selectbox("Page", page_numbers, key="preview-page")
    page = next((item for item in pages if item.page_number == selected_page), None)
    if page:
        st.text_area("Extracted text", page.text[:4000], height=260)

    st.divider()
    render_document_profiles(engine)
    render_evaluation_panel(engine)
    st.markdown("</div>", unsafe_allow_html=True)


def render_semantic_search(engine: RagEngine, selected_docs: list[str]) -> None:
    st.subheader("Semantic Search")
    query = st.text_input("Search passages without generating an answer", key="semantic-search-query")
    top_k = st.slider("Search results", 3, 20, 8, key="semantic-search-topk")
    if not query:
        st.info("Enter a search query to inspect the most relevant passages directly.")
        return
    results = engine.semantic_search(query, document_names=selected_docs, top_k=top_k)
    for item in results:
        chunk = item.chunk
        score = max(0.0, min(1.0, item.score))
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{html.escape(chunk.document_name)}</div>
                <div class="source-meta">Page {chunk.page_number} | score {score:.3f}</div>
                <div class="score-bar"><div class="score-fill" style="width:{int(score * 100)}%"></div></div>
                <div class="source-text">{html.escape(shorten(chunk.text.replace(chr(10), " "), width=520, placeholder="..."))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_advanced_insights(engine: RagEngine) -> None:
    st.subheader("Document Intelligence")
    profile_tab, risk_tab, timeline_tab, graph_tab, privacy_tab = st.tabs(
        ["Profiles", "Risks & Actions", "Timeline", "Knowledge Graph", "Privacy"]
    )
    with profile_tab:
        render_document_profiles(engine)
    with risk_tab:
        for profile in engine.profiles:
            st.markdown(f"### {profile.document_name}")
            if profile.clauses:
                st.markdown("**Detected clauses:** " + ", ".join(profile.clauses))
            if profile.risks:
                st.markdown("**Risk evidence**")
                for risk in profile.risks:
                    st.markdown(f"- {risk}")
            if profile.action_items:
                st.markdown("**Action items / obligations**")
                for item in profile.action_items:
                    st.markdown(f"- {item}")
            if not profile.risks and not profile.action_items and not profile.clauses:
                st.caption("No obvious risk, clause, or action patterns detected.")
    with timeline_tab:
        events = build_timeline(engine.pages)
        if not events:
            st.caption("No date-linked events detected.")
        for event in events:
            st.markdown(
                f"- **{event['date']}** — {event['event']} "
                f"({event['document']}, page {event['page']})"
            )
    with graph_tab:
        edges = build_knowledge_graph(engine.profiles)
        if not edges:
            st.caption("No graph relationships detected.")
        for edge in edges:
            st.markdown(
                f"- `{edge['source']}` **{edge['relation']}** `{edge['target']}`"
            )
    with privacy_tab:
        any_pii = False
        for profile in engine.profiles:
            if profile.pii:
                any_pii = True
                st.warning(f"{profile.document_name}: possible {', '.join(profile.pii)} detected")
        if not any_pii:
            st.success("No common PII patterns detected in the extracted text.")
        st.caption("Cloud prompts redact common emails, phone numbers, SSN-like values, and account-like values.")


def render_feedback_controls(question: str, answer: str) -> None:
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Useful", key=f"feedback-good-{len(st.session_state.get('answer_records', []))}"):
        save_feedback(question, answer, "useful")
        st.toast("Feedback saved")
    if col_b.button("Missing source", key=f"feedback-source-{len(st.session_state.get('answer_records', []))}"):
        save_feedback(question, answer, "missing_source")
        st.toast("Feedback saved")
    if col_c.button("Wrong", key=f"feedback-bad-{len(st.session_state.get('answer_records', []))}"):
        save_feedback(question, answer, "wrong")
        st.toast("Feedback saved")


def save_feedback(question: str, answer: str, label: str) -> None:
    import json
    from datetime import datetime

    path = Path("data/feedback.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "question": question,
        "answer": answer,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def render_document_profiles(engine: RagEngine) -> None:
    with st.expander("Document analysis", expanded=False):
        cards = []
        for profile in engine.profiles:
            name = html.escape(profile.document_name)
            doc_type = html.escape(profile.document_type)
            summary = html.escape(profile.summary)
            topics = html.escape(", ".join(profile.key_topics[:6]) or "No dominant topics detected")
            dates = html.escape(", ".join(profile.dates[:4]) or "No dates detected")
            entities = html.escape(", ".join(profile.entities[:4]) or "No names detected")
            cards.append(
                f"""
                <div class="file-card">
                    <div class="file-card-title">{name}</div>
                    <div class="file-card-meta">{doc_type} | {profile.pages} page(s) | {profile.chunks} chunk(s)</div>
                    <div class="source-text">{summary}</div>
                    <div class="source-meta">Topics: {topics}</div>
                    <div class="source-meta">Dates: {dates}</div>
                    <div class="source-meta">Names: {entities}</div>
                </div>
                """
            )
        st.markdown(f"<div class='file-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_suggested_questions(engine: RagEngine) -> None:
    questions = []
    for profile in engine.profiles:
        questions.extend(profile.suggested_questions[:3])
    questions.extend(
        [
            "Compare the uploaded documents",
            "Give me a concise executive summary",
            "List the strongest evidence with page references",
        ]
    )
    deduped = []
    for question in questions:
        if question not in deduped:
            deduped.append(question)
    st.caption("Suggested questions")
    cols = st.columns(3)
    for index, question in enumerate(deduped[:6]):
        with cols[index % 3]:
            if st.button(question, key=f"suggested-{index}"):
                st.session_state["pending_question"] = question


def render_evaluation_panel(engine: RagEngine) -> None:
    with st.expander("Retrieval diagnostics", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Documents", len(engine.document_names))
        col_b.metric("Pages", len(engine.pages))
        col_c.metric("Chunks", len(engine.chunks))
        col_d.metric("Embedder", engine.embedder.model_name)

        records = st.session_state.get("answer_records", [])
        if records:
            latest = records[-1]
            st.caption(
                f"Last answer: {latest['elapsed']:.2f}s, confidence {latest['confidence']:.3f}, "
                f"{latest['source_count']} source chunk(s)."
            )


def render_chat(
    provider: str,
    selected_docs: list[str] | None,
    answer_style: str,
    persona: str,
    rerank: bool,
) -> None:
    st.subheader("Query Console")
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input("Ask a question about the uploaded documents")
    question = st.session_state.pop("pending_question", None) or typed_question
    if not question:
        render_report_download()
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    engine: RagEngine | None = st.session_state.get("engine")
    if engine is None:
        response_text = "Please upload files first. I will analyze them, then you can ask questions about their contents."
        messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
        render_report_download()
        return

    history = [ChatTurn(role=m["role"], content=m["content"]) for m in messages[:-1]]
    retrieval_question = apply_answer_style(question, answer_style, persona)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and drafting an answer..."):
            start = time.perf_counter()
            answer = engine.ask(
                retrieval_question,
                history=history,
                provider=provider,
                document_names=selected_docs,
                rerank=rerank,
            )
            elapsed = time.perf_counter() - start
        st.markdown(answer.text)
        if answer.sources:
            render_citation_badges(answer.sources)
            render_sources(answer.sources)
        st.caption(f"Answered in {elapsed:.2f}s using {answer.used_provider}. Confidence {answer.confidence:.3f}.")
        st.download_button(
            "Export this answer",
            build_single_answer_report(question, answer.text, answer.sources),
            file_name="ask-my-pdf-answer.md",
            mime="text/markdown",
            key=f"answer-export-{len(messages)}",
        )
        render_feedback_controls(question, answer.text)

    messages.append({"role": "assistant", "content": answer.text})
    st.session_state.setdefault("answer_records", []).append(
        {
            "question": question,
            "answer": answer.text,
            "elapsed": elapsed,
            "confidence": answer.confidence,
            "provider": answer.used_provider,
            "source_count": len(answer.sources),
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
    )
    render_report_download()


def apply_answer_style(question: str, answer_style: str, persona: str) -> str:
    style_notes = {
        "Concise": "Answer concisely in 3-5 sentences.",
        "Detailed": "Answer in detail and include important caveats.",
        "Bullet points": "Answer using clear bullet points.",
        "Simple explanation": "Explain this simply for a beginner.",
    }
    note = style_notes.get(answer_style)
    persona_note = f"Answer as a {persona.lower()}."
    style_note = f" Preferred answer style: {note}" if note else ""
    return f"{question}\n\n{persona_note}{style_note}"


def render_citation_badges(sources) -> None:
    seen = set()
    badges = []
    for item in sources:
        key = (item.chunk.document_name, item.chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        label = html.escape(f"{item.chunk.document_name} | p.{item.chunk.page_number}")
        badges.append(f"<span class='citation-badge'>{label}</span>")
    if badges:
        st.markdown(f"<div class='citation-row'>{''.join(badges)}</div>", unsafe_allow_html=True)


def build_single_answer_report(question: str, answer: str, sources) -> str:
    lines = ["# Ask My PDF Bot Answer", "", "## Question", question, "", "## Answer", answer, "", "## Sources"]
    for item in sources:
        lines.extend(
            [
                f"- {item.chunk.document_name}, page {item.chunk.page_number}, score {item.score:.3f}",
                f"  {item.chunk.text[:900]}",
            ]
        )
    return "\n".join(lines)


def render_sources(sources) -> None:
    with st.expander("Sources", expanded=True):
        for item in sources:
            chunk = item.chunk
            document_name = html.escape(chunk.document_name)
            excerpt = html.escape(shorten(chunk.text.replace("\n", " "), width=520, placeholder="..."))
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{document_name}</div>
                    <div class="source-meta">Page {chunk.page_number} | score {item.score:.3f} | semantic {item.semantic_score:.3f} | keyword {item.keyword_score:.3f}</div>
                    <div class="source-text">{excerpt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_report_download() -> None:
    messages = st.session_state.get("messages", [])
    records = st.session_state.get("answer_records", [])
    if not messages:
        return
    report = build_chat_report(messages, records)
    st.download_button(
        "Download chat report",
        report,
        file_name="ask-my-pdf-report.md",
        mime="text/markdown",
    )


def build_chat_report(messages: list[dict[str, str]], records: list[dict]) -> str:
    engine: RagEngine | None = st.session_state.get("engine")
    lines = ["# Ask My PDF Bot Report", ""]
    if engine:
        lines.extend(
            [
                "## Corpus",
                f"- Documents: {len(engine.document_names)}",
                f"- Pages: {len(engine.pages)}",
                f"- Chunks: {len(engine.chunks)}",
                f"- Embedder: {engine.embedder.model_name}",
                "",
            ]
        )
    lines.append("## Chat")
    for message in messages:
        lines.extend([f"### {message['role'].title()}", message["content"], ""])
    if records:
        lines.append("## Retrieved Sources")
        for record in records:
            lines.extend([f"### Question: {record['question']}", f"Confidence: {record['confidence']:.3f}", ""])
            for source in record["sources"]:
                lines.extend(
                    [
                        f"- {source['document']}, page {source['page']}, score {source['score']:.3f}",
                        f"  {source['text']}",
                    ]
                )
            lines.append("")
    return "\n".join(lines)


def upload_signature(payload: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha1()
    for name, content in sorted(payload, key=lambda item: item[0]):
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(hashlib.sha1(content).digest())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
