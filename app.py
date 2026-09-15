import streamlit as st
import os
import re
import numpy as np
from io import BytesIO

import faiss
from pypdf import PdfReader
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — dark dashboard feel
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d27 0%, #12151f 100%);
        border-right: 1px solid #2d3748;
    }
    .user-msg {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px auto;
        max-width: 78%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    }
    .bot-msg {
        background: linear-gradient(135deg, #1e2530 0%, #252d3d 100%);
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px auto 8px 0;
        max-width: 78%;
        border: 1px solid #2d3748;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .source-tag {
        background: #1a3a5c;
        color: #60a5fa;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 8px;
        display: inline-block;
        border: 1px solid #2563eb44;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2530, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-num { font-size: 2rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    .status-ready {
        background: #052e16; color: #4ade80;
        border: 1px solid #166534; padding: 6px 14px;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    }
    .status-idle {
        background: #1c1f2e; color: #94a3b8;
        border: 1px solid #374151; padding: 6px 14px;
        border-radius: 20px; font-size: 0.82rem;
    }
    .stTextInput input {
        background-color: #1e2530 !important; color: #e2e8f0 !important;
        border: 1px solid #374151 !important; border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
    }
    .section-header {
        color: #60a5fa; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 8px 0 4px 0; border-bottom: 1px solid #2d3748; margin-bottom: 12px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 6
PAGE_NEARBY = 1
GEMINI_MODEL = "gemini-3.6-flash"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
defaults = {
    "api_key"      : "",
    "chat_history" : [],
    "vector_store" : None,
    "pdf_name"     : "",
    "pdf_pages"    : 0,
    "pdf_chunks"   : 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────

def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks without creating unnecessary chunks."""
    chunks = []
    start = 0
    step = max(1, size - overlap)

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks


def extract_pdf(pdf_bytes: bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append((i + 1, txt))
    return pages, len(reader.pages)


@st.cache_resource
def get_embedding_model():
    """Load the local embedding model once per Streamlit process."""
    return SentenceTransformer(LOCAL_EMBED_MODEL)


def embed_texts(texts: list, client=None) -> np.ndarray:
    """Create embeddings locally; Gemini is not used for PDF embeddings."""
    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")

def build_vector_store(pdf_bytes: bytes, client=None):
    pages, total_pages = extract_pdf(pdf_bytes)
    chunks, meta = [], []

    for page_num, page_text in pages:
        for c in chunk_text(page_text):
            chunks.append(c)
            meta.append({"page": page_num})

    if not chunks:
        return None, 0, 0

    with st.spinner(f"🔢 Creating local embeddings for {len(chunks)} chunks…"):
        embeddings = embed_texts(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {"index": index, "chunks": chunks, "meta": meta, "total_pages": total_pages}, total_pages, len(chunks)


def is_overview_question(query: str) -> bool:
    q = query.lower().strip()
    phrases = (
        "what is this document about",
        "what is the document about",
        "what is this pdf about",
        "what is the pdf about",
        "summarize the document",
        "summarise the document",
        "summarize this document",
        "summarise this document",
        "give me a summary",
        "give me the summary",
        "summarize the key points",
        "summarise the key points",
        "main findings",
        "main conclusions",
        "key points of the document",
    )
    return any(p in q for p in phrases)


def extract_page_request(query: str):
    """
    Detect explicit page references such as:
    'page 200', 'on page 200', 'near page 200', 'pages 200-205'.
    Returns a tuple describing the requested page range, or None.
    """
    q = query.lower()

    range_match = re.search(r"pages?\s*(\d+)\s*(?:-|to)\s*(\d+)", q)
    if range_match:
        first = int(range_match.group(1))
        last = int(range_match.group(2))
        return ("range", min(first, last), max(first, last))

    page_match = re.search(r"page\s*(?:number\s*)?(\d+)", q)
    if page_match:
        page = int(page_match.group(1))

        if re.search(r"\bnear\b|\baround\b|\bclose to\b", q):
            return ("near", page, page)

        return ("exact", page, page)

    return None


def retrieve_by_page(query: str, vector_store: dict, top_k=TOP_K):
    """Retrieve directly from page metadata when the user specifies pages."""
    request = extract_page_request(query)

    if not request:
        return None

    mode, first_page, last_page = request
    total_pages = vector_store.get("total_pages", 0)

    if total_pages:
        first_page = max(1, min(first_page, total_pages))
        last_page = max(1, min(last_page, total_pages))

    if mode == "near":
        first_page = max(1, first_page - PAGE_NEARBY)
        last_page = min(total_pages or last_page + PAGE_NEARBY,
                        last_page + PAGE_NEARBY)

    matching = []

    for idx, meta in enumerate(vector_store["meta"]):
        page = meta["page"]
        if first_page <= page <= last_page:
            matching.append({
                "text": vector_store["chunks"][idx],
                "page": page,
                "score": 1.0,
            })

    # For an exact page request, return every chunk from that page.
    # For a range/near request, cap the amount of context sent to Gemini.
    if mode == "exact":
        return matching

    return matching[:max(top_k * 2, 10)]


def semantic_retrieve(query: str, vector_store: dict, top_k=TOP_K):
    q_emb = embed_texts([query])

    scores, idxs = vector_store["index"].search(
        q_emb,
        min(top_k, len(vector_store["chunks"]))
    )

    results = []

    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0:
            results.append({
                "text": vector_store["chunks"][idx],
                "page": vector_store["meta"][idx]["page"],
                "score": float(score),
            })

    return results


def retrieve(query: str, vector_store: dict, client=None, top_k=TOP_K):
    """
    Route page-number questions to page metadata and all other questions
    to semantic FAISS retrieval.
    """
    page_results = retrieve_by_page(query, vector_store, top_k)

    if page_results is not None:
        return page_results

    results = semantic_retrieve(query, vector_store, top_k)

    if is_overview_question(query):
        seen = {id(result) for result in results}
        total = len(vector_store["chunks"])

        if total > 0:
            representative_count = min(8, total)
            representative_idxs = np.linspace(
                0, total - 1, representative_count, dtype=int
            )

            for raw_idx in representative_idxs:
                idx = int(raw_idx)
                candidate = {
                    "text": vector_store["chunks"][idx],
                    "page": vector_store["meta"][idx]["page"],
                    "score": None,
                }

                if not any(
                    item["text"] == candidate["text"] for item in results
                ):
                    results.append(candidate)

    return results


def build_prompt(query: str, context_chunks: list, pdf_name: str) -> str:
    ctx = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in context_chunks
    )

    page_request = extract_page_request(query)

    if page_request:
        if page_request[0] == "exact":
            page_instruction = (
                f"The user explicitly asked about page {page_request[1]}. "
                f"Use ONLY the chunks labeled Page {page_request[1]} and "
                "answer from that page."
            )
        else:
            page_instruction = (
                "The user explicitly asked about a page or page range. "
                "Use the supplied page-labeled chunks and identify the "
                "relevant page numbers in the answer."
            )
    else:
        page_instruction = ""

    overview_instruction = (
        "For this broad overview question, synthesize the supplied excerpts "
        "from different parts of the document to explain its topic, purpose, "
        "main themes, findings, or conclusions. Do not require one exact "
        "sentence to answer the question."
        if is_overview_question(query)
        else
        "Answer the user's specific question using the supplied context."
    )

    if page_instruction:
        overview_instruction = page_instruction

    return f"""You are a precise document assistant. Your ONLY knowledge source is the PDF: "{pdf_name}".

STRICT RULES:
1. {overview_instruction}
2. Never use outside knowledge.
3. For a page-specific question, answer from the page-labeled context supplied to you.
4. If the supplied context genuinely does not contain enough information, respond: "I couldn't find information about that in the uploaded PDF."
5. Cite page numbers inline whenever you use information from the document, e.g. (Page 3).
6. Do not invent facts, names, numbers, or conclusions.
7. Keep the answer clear, concise, and useful.

=== DOCUMENT CONTEXT ===
{ctx}
=== END DOCUMENT CONTEXT ===

USER QUESTION: {query}

ANSWER:"""

def ask_gemini(prompt: str, client) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.05,
            max_output_tokens=1024,
        ),
    )
    return response.text


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF RAG Chatbot")
    st.markdown(
        "<div style='color:#64748b;font-size:0.82rem;margin-bottom:16px;'>"
        "Local Embeddings + Google Gemini + FAISS + Page-Aware Search</div>",
        unsafe_allow_html=True,
    )

    # API Key
    st.markdown("<div class='section-header'>🔑 API Configuration</div>", unsafe_allow_html=True)

    try:
        secret_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        secret_api_key = ""

    if secret_api_key:
        st.session_state.api_key = secret_api_key
        st.success("✅ API configured", icon="🔑")
    else:
        api_input = st.text_input(
            "Google AI Studio API Key",
            type="password",
            placeholder="AIza…",
            value=st.session_state.api_key,
            help="Enter your Gemini API key",
        )
        if api_input != st.session_state.api_key:
            st.session_state.api_key = api_input

    if not st.session_state.api_key:
        st.warning("Enter your Gemini API key to start", icon="⚠️")

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    st.markdown("<div class='section-header'>📤 Upload Document</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        if st.button("⚙️ Process PDF", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Enter your API key first.")
            else:
                try:
                    client = get_client(st.session_state.api_key)
                    pdf_bytes = uploaded_file.read()
                    store, pages, chunks = build_vector_store(pdf_bytes, client)
                    if store:
                        st.session_state.vector_store = store
                        st.session_state.pdf_name     = uploaded_file.name
                        st.session_state.pdf_pages    = pages
                        st.session_state.pdf_chunks   = chunks
                        st.session_state.chat_history = []
                        st.success(f"✅ Indexed {chunks} chunks from {pages} pages!")
                    else:
                        st.error("No text found in PDF. Try another file.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Document Stats
    if st.session_state.vector_store:
        st.markdown("<br><div class='section-header'>📊 Document Stats</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-num'>{st.session_state.pdf_pages}</div>
                <div class='metric-label'>Pages</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-num'>{st.session_state.pdf_chunks}</div>
                <div class='metric-label'>Chunks</div></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#1e2530;border:1px solid #2d3748;border-radius:8px;
                    padding:10px;margin-top:8px;'>
            <div style='color:#94a3b8;font-size:0.75rem;'>Active Document</div>
            <div style='color:#60a5fa;font-size:0.85rem;font-weight:600;
                        word-break:break-all;'>{st.session_state.pdf_name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#475569;font-size:0.75rem;line-height:1.6;'>
    <b style='color:#60a5fa;'>How it works</b><br>
    1. Enter your Gemini API key<br>
    2. Upload a PDF document<br>
    3. Click "Process PDF"<br>
    4. Ask questions — answers come<br>
       strictly from your document
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN — HEADER
# ──────────────────────────────────────────────
h_col, s_col = st.columns([3, 1])
with h_col:
    st.markdown("## 💬 Chat with your PDF")
with s_col:
    if st.session_state.vector_store:
        st.markdown("<br><span class='status-ready'>● Ready</span>", unsafe_allow_html=True)
    else:
        st.markdown("<br><span class='status-idle'>○ No document</span>", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
# CHAT DISPLAY
# ──────────────────────────────────────────────
if not st.session_state.chat_history:
    if st.session_state.vector_store:
        st.markdown(f"""
        <div style='text-align:center;padding:40px;color:#475569;'>
            <div style='font-size:2.5rem;'>🎯</div>
            <div style='font-size:1.1rem;color:#60a5fa;font-weight:600;margin-top:8px;'>
                Document ready!</div>
            <div style='font-size:0.9rem;margin-top:8px;'>
                Ask anything about <b>{st.session_state.pdf_name}</b></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#475569;'>
            <div style='font-size:3rem;'>📄</div>
            <div style='font-size:1.1rem;color:#60a5fa;font-weight:600;margin-top:12px;'>
                Upload a PDF to get started</div>
            <div style='font-size:0.88rem;margin-top:8px;'>
                Use the sidebar to upload and process your document</div>
        </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-msg'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            sources_html = ""
            if msg.get("sources"):
                unique_pages = sorted(set(s["page"] for s in msg["sources"]))
                sources_html = "<div style='margin-top:10px;'>"
                for p in unique_pages:
                    sources_html += f"<span class='source-tag'>📄 Page {p}</span>"
                sources_html += "</div>"
            st.markdown(
                f"<div class='bot-msg'>🤖 {msg['content']}{sources_html}</div>",
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────
# INPUT ROW
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
i_col, b_col = st.columns([5, 1])
with i_col:
    user_query = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the main findings of this paper?",
        label_visibility="collapsed",
        key="query_input",
        disabled=not st.session_state.vector_store,
    )
with b_col:
    send = st.button(
        "Send ➤",
        use_container_width=True,
        disabled=not st.session_state.vector_store,
    )

# ──────────────────────────────────────────────
# PROCESS QUERY
# ──────────────────────────────────────────────
if send and user_query.strip():
    if not st.session_state.api_key:
        st.error("Please enter your Google Gemini API key in the sidebar.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query.strip(), "sources": []}
        )
        client = get_client(st.session_state.api_key)
        context = []

        with st.spinner("🔍 Searching document…"):
            context = retrieve(user_query, st.session_state.vector_store, client)

        if not context:
            answer  = "I couldn't find information about that in the uploaded PDF. Please try rephrasing or check that this topic is covered."
            context = []
        else:
            with st.spinner("🤖 Generating answer with Gemini…"):
                try:
                    prompt = build_prompt(user_query, context, st.session_state.pdf_name)
                    answer = ask_gemini(prompt, client)
                except Exception as e:
                    answer  = f"⚠️ API error: {e}"
                    context = []

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": context}
        )
        st.rerun()

# ──────────────────────────────────────────────
# QUICK PROMPTS (shown when doc ready, chat empty)
# ──────────────────────────────────────────────
if st.session_state.vector_store and not st.session_state.chat_history:
    st.markdown(
        "<div class='section-header' style='margin-top:20px;'>💡 Quick Questions</div>",
        unsafe_allow_html=True,
    )
    q1, q2, q3 = st.columns(3)
    quick_qs = [
        "What is this document about?",
        "Summarize the key points",
        "What are the main conclusions?",
    ]
    for col, q in zip([q1, q2, q3], quick_qs):
        with col:
            if st.button(q, use_container_width=True, key=f"qq_{q}"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": q, "sources": []}
                )
                client  = get_client(st.session_state.api_key)
                context = retrieve(q, st.session_state.vector_store, client)
                if context:
                    prompt = build_prompt(q, context, st.session_state.pdf_name)
                    try:
                        answer = ask_gemini(prompt, client)
                    except Exception as e:
                        answer, context = f"⚠️ {e}", []
                else:
                    answer, context = "I couldn't find relevant information in the PDF.", []
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": context}
                )
                st.rerun()
