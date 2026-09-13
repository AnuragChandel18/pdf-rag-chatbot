# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its content. The application retrieves the most relevant sections from the uploaded document and uses Google Gemini to generate answers based on the retrieved context.

## 🚀 Live Demo

**Streamlit App:**
[https://pdf-rag-chatbot-36yadsnlsx9fa4dzxch5e8.streamlit.app/]

## 📌 Project Overview

This project demonstrates how a PDF-based question-answering system can be built using **Retrieval-Augmented Generation (RAG)**.

Instead of sending the entire PDF to the language model, the application:

1. Extracts text from the uploaded PDF.
2. Splits the text into smaller overlapping chunks.
3. Converts the chunks into vector embeddings using `gemini-embedding-001`.
4. Stores the embeddings in a FAISS vector index.
5. Converts the user's question into an embedding.
6. Retrieves the most relevant document chunks.
7. Sends the retrieved context to Google Gemini.
8. Generates a response based on the relevant PDF content.

## ✨ Features

* 📤 Upload PDF documents
* 📑 Extract text from PDF files
* ✂️ Text chunking with overlap
* 🔢 Gemini text embeddings
* 🔎 Semantic similarity search using FAISS
* 🤖 Gemini-powered answer generation
* 💬 Ask multiple questions about the uploaded document
* ☁️ Deployed using Streamlit Community Cloud

## 🛠️ Tech Stack

| Technology             | Purpose                            |
| ---------------------- | ---------------------------------- |
| Python                 | Application development            |
| Streamlit              | Web application interface          |
| Google Gemini          | Embeddings and response generation |
| `gemini-embedding-001` | Text embeddings                    |
| FAISS                  | Vector similarity search           |
| PyPDF                  | PDF text extraction                |
| NumPy                  | Numerical and vector operations    |

## 🧠 RAG Architecture

```text
                 PDF Upload
                     │
                     ▼
              PDF Text Extraction
                     │
                     ▼
                Text Chunking
              (600 chars / overlap)
                     │
                     ▼
            Gemini Embeddings
                     │
                     ▼
                FAISS Index
                     │
                     │
User Question ───────┘
      │
      ▼
Question Embedding
      │
      ▼
Similarity Search
      │
      ▼
Top Relevant Chunks
      │
      ▼
   Google Gemini
      │
      ▼
   Final Answer
```

## 📂 Project Structure

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> `.env` is excluded from Git to prevent API credentials from being committed.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AnuragChandel18/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file and add your Gemini API key.

```text
GEMINI_API_KEY=your_api_key_here
```

For Streamlit Cloud deployment, configure the API key through Streamlit Secrets instead of committing it to the repository.

### 6. Run the application

```bash
streamlit run app.py
```

## 🔐 Security

API credentials are not stored in the GitHub repository.

The `.gitignore` file excludes:

```text
.env
venv/
.venv/
__pycache__/
*.pyc
```

## 🎯 What I Learned

Through this project, I worked with:

* Retrieval-Augmented Generation (RAG)
* Text chunking and document processing
* Vector embeddings
* Semantic similarity search
* FAISS vector databases
* Gemini API integration
* Streamlit application development
* Deploying an AI application to the cloud
* Managing API credentials securely

## 👨‍💻 Author

**Anurag Singh Chandel**

B.Tech Computer Science Engineering

Interested in **Data Analytics, Python, SQL, and Generative AI**.

---

⭐ If you find this project useful, consider giving it a star.
