# 📄 PDF RAG Chatbot
**Google Gemini 2.0 Flash + FAISS | Streamlit Dashboard**

A fully local RAG chatbot that answers questions strictly from your uploaded PDF.

## ⚡ Quick Start

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## 🔑 Get a Free API Key
→ https://aistudio.google.com/apikey  (free, no credit card needed)

## 🏗️ RAG Architecture

```
PDF Upload
   ↓
Text Extraction (pypdf, page-by-page)
   ↓
Chunking (600 chars, 100 overlap)
   ↓
Gemini Embeddings (text-embedding-004, 768-dim)
   ↓
FAISS Vector Index (cosine similarity)
   ↓
User Query → Query Embedding → Top-5 Retrieval
                                    ↓
                         Prompt + Context → Gemini 2.0 Flash → Answer
```

## ✨ Features
- **Strict RAG** — answers only from your PDF, never hallucinates
- **Page citations** — every answer cites source pages
- **Gemini embeddings** — no local model download needed
- **Quick prompts** — one-click starter questions
- **Dark dashboard UI** — clean, professional look
- **Multi-chunk retrieval** — top-5 relevant chunks per query

## 📁 Project Structure
```
pdf_rag_chatbot/
├── app.py            ← Main Streamlit app (single file)
├── requirements.txt  ← Python dependencies
└── README.md
```
