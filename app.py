import streamlit as st
import os
import json
import numpy as np
import urllib.request
import urllib.error
from io import BytesIO

import faiss
from pypdf import PdfReader
from google import genai
from google.genai import types

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
CHUNK_SIZE   = 600   # chars per chunk
CHUNK_OVERLAP= 100
TOP_K        = 5
GEMINI_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"   # Gemini embedding model (no models/ prefix)
EMBED_DIM    = 768

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
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        start += size - overlap
    return chunks


def extract_pdf(pdf_bytes: bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append((i + 1, txt))
    return pages, len(reader.pages)


def embed_texts(texts: list, api_key: str) -> np.ndarray:
    """
    Create Gemini embeddings using the Gemini API REST endpoint.
    The API key is sent securely in the x-goog-api-key header.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:batchEmbedContents"
    )
    all_embs = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({
            "requests": [
                {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Embedding API error {e.code}: {e.read().decode()}") from e
        for emb_obj in data["embeddings"]:
            all_embs.append(emb_obj["values"])
    return np.array(all_embs, dtype="float32")


def build_vector_store(pdf_bytes: bytes, api_key: str):
    pages, total_pages = extract_pdf(pdf_bytes)
    chunks, meta = [], []
    for page_num, page_text in pages:
        for c in chunk_text(page_text):
            chunks.append(c)
            meta.append({"page": page_num})

    if not chunks:
        return None, 0, 0

    with st.spinner(f"🔢 Embedding {len(chunks)} chunks with Gemini…"):
        embeddings = embed_texts(chunks, api_key)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {"index": index, "chunks": chunks, "meta": meta}, total_pages, len(chunks)


def retrieve(query: str, vector_store: dict, api_key: str, top_k=TOP_K):
    q_emb = embed_texts([query], api_key)
    faiss.normalize_L2(q_emb)
    scores, idxs = vector_store["index"].search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0 and score > 0.12:
            results.append({
                "text" : vector_store["chunks"][idx],
                "page" : vector_store["meta"][idx]["page"],
                "score": float(score),
            })
    return results


def build_prompt(query: str, context_chunks: list, pdf_name: str) -> str:
    ctx = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are a precise document assistant. Your ONLY knowledge source is the PDF: "{pdf_name}".

STRICT RULES:
1. Answer ONLY from the context provided below — never use outside knowledge.
2. If the answer is not in the context, respond: "I couldn't find information about that in the uploaded PDF."
3. Cite page numbers inline when referencing information, e.g. (Page 3).
4. Be concise, factual, and structured.

=== DOCUMENT CONTEXT ===
{ctx}
=== END CONTEXT ===

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
        "Powered by Google Gemini + FAISS</div>",
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
                    store, pages, chunks = build_vector_store(pdf_bytes, st.session_state.api_key)
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
            context = retrieve(user_query, st.session_state.vector_store, st.session_state.api_key)

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
                context = retrieve(q, st.session_state.vector_store, st.session_state.api_key)
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
