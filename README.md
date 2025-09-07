# 🚀 Building a RAG-Powered Web Content Question-Answering System with Citation Support

---

## 🎯 Problem Statement
Website Retrieval Assignment - Developed a Retrieval Augmented Generation (RAG) system to answer questions based on content from specific websites.
- 🔎 Retrieves relevant, up-to-date content from specific websites  
- 🧠 Combines that with **LLM (Large Language Model) generation**  
- 📑 Provides **citations (links back to the source)** for transparency  
- 🔐 Allows **user authentication** before use  

---
### 📂 Project Structure
```bash
RAG-QA-APP/
│
├── backend/                     # Backend service (FastAPI)
│   ├── __init__.py              # Marks package, can include initialization code
│   ├── auth.py                  # Handles authentication (JWT, user auth, etc.)
│   ├── embeddings.py            # Embedding generation logic
│   ├── engine.py                # Core RAG engine (retrieval + generation)
│   ├── main.py                  # FastAPI entry point
│   ├── models.py                # Pydantic models / schema definitions
│   ├── router.py                # API routes definition
│   ├── vector_store.py          # Vector database operations (Chroma)
│   └── web_loader.py            # Logic to fetch and preprocess website content
│
├── frontend/                    # Frontend service (Streamlit)
│   ├── app.py                   # Frontend app entry point
│
├── .env                         # Environment variables (API keys, configs)
├── requirements.txt             # Python dependencies

```
---
## 🧩 Architecture Overview

### 🔹 1. Data Source & Indexing
- Users supply website URLs via the **Streamlit frontend**.  
- **`web_loader.py`** → Fetches HTML content (BeautifulSoup + Readability).  
- **`engine.py`**:
  - Splits content into chunks (LangChain `RecursiveCharacterTextSplitter`).  
  - Generates embeddings using either:
    - **OpenAI embeddings** (if `OPENAI_API_KEY` set), or  
    - **SentenceTransformers / HuggingFace models** (offline).  
  - Stores chunks + embeddings in **ChromaDB** (persistent vector DB).  

👉 Supports **Replace** mode (wipe & re-index).

---

### 🔹 2. Vector Database
- **`vector_store.py`** manages **Chroma collections**:
  - Stores text chunks, embeddings, and metadata (`url`, `title`, `chunk_id`, etc.).  
  - Supports **reset, add, query, and metadata fetch**.  
- Retrieval uses **cosine similarity** to find the most relevant chunks.  

---

### 🔹 3. Retrieval-Augmented Generation (RAG)
- **`engine.py → retrieve()`**:
  - Encodes user query into embeddings.  
  - Finds top-K relevant document chunks.  
- **`router.py`**:
  - Handles `/api/v1/chat`.  
  - Calls engine’s retrieval.  
  - Uses **LangChain’s RetrievalQA + LLM** for polished answers.  
  - Attaches **citations**: snippet text, URL, and title.  

---

### 🔹 4. API Layer (Backend - FastAPI)
- **`main.py`** → initializes FastAPI app.  
- **`router.py`** → defines endpoints:  
  - `POST /api/v1/login` → JWT authentication  
  - `POST /api/v1/index` → Index new URLs  
  - `GET  /api/v1/index` → List indexed URLs  
  - `POST /api/v1/chat` → Ask a question, get RAG-powered answer with citations  
- **`auth.py`** → Implements JWT-based authentication (with expiration).  

---

### 🔹 5. Frontend (Streamlit)
- **`app.py`** provides a **multi-step interface**:
  1. Login (JWT-protected)   
  3. Indexing Page (enter URLs → index into Chroma)  
  4. Chat Page (ask questions → get answers + citations)  

- Answers appear in **highlighted cards**.  
- Citations show **exact sources inline** after the content.  

---

### 🔹 6. Authentication
- User must **login first** with credentials from `.env`.  
- JWT tokens are issued by **FastAPI backend** and stored in session.  
- All API requests from Streamlit include the token in headers.  

---

## 🛠️ Tech Stack

| Layer        | Technology                               |
|--------------|------------------------------------------|
| Frontend     | Streamlit                                |
| Backend      | FastAPI                                  |
| Embeddings   | OpenAI API / SentenceTransformers        |
| Vector DB    | ChromaDB                                 |
| Orchestration| LangChain                                |
| Auth         | JWT (`python-jose`, `PyJWT`)             |
| Scraping     | Requests + BeautifulSoup + Readability   |
| Deployment   | Local / Cloud (AWS, GCP, Heroku)         |

---

## 🔍 Business Use Cases
- Question-answering systems for company documentation/knowledge bases
- Automated customer support using website content
- Information retrieval from multiple web sources
- Content-aware chatbots with citation capabilities

---

## 📊 Workflow Example

1. User logs in with **username/password**.  
2. Enters URLs → system fetches and indexes content.  
3. User asks: *“What is Retrieval-Augmented Generation?”*  
4. Backend:  
   - Embeds query → finds matching chunks from indexed sites.  
   - LLM generates coherent summary.  
   - Citations are attached.  
5. Streamlit UI shows answer:  

> **Answer**  
> Retrieval-Augmented Generation (RAG) combines retrieval from external data with generation using large language models, ensuring accurate and grounded responses.  
>  
> **From [huyenchip.com](https://huyenchip.com/2024/07/25/genai-platform.html):**  
> “This is what the overall architecture looks like…”  

---

## 🚀 Getting Started

### 1️⃣ Clone & Install
```bash
git clone https://github.com/SSaranya19/RAG-Website-Retrieval-QA.git
cd RAG-Website-Retrieval-QA
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
Copy .env.example → .env and set values.
Run Backend : uvicorn backend.main:app --reload --port 8000
Run Frontend: streamlit run app.py
