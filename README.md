# 🎓 PGP AI Course Advisor
### AI-Powered Course Assistant for Program Discovery & Query Resolution
> Built for **FrostHack @ Xpecto'26, IIT Mandi** | Team: Team Paranox (Newton School of Technology, Rishihood University)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Problem Statement

Business schools face difficulty addressing large volumes of prospective student queries about curriculum, fees, admissions, and career outcomes. This assistant solves that with a **RAG-powered AI chatbot** grounded in official program data — no hallucinations, no guesswork.

---

## ✨ Features

- 🔍 **RAG Pipeline** — Retrieval-Augmented Generation with ChromaDB vector store
- 🌐 **Language** — Supports English
- 📄 **Multi-source ingestion** — Website (JS-rendered via Jina Reader) + PDF brochure
- 🛡️ **Anti-hallucination** — Answers strictly grounded in official program data
- 💡 **Agentic follow-ups** — Auto-suggests 3 related questions after every answer
- ⚡ **Fast inference** — LLaMA 3.3 70B via Groq API (free tier)
- 🎨 **Premium dark UI** — Masters' Union branded Streamlit interface

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      DATA SOURCES                        │
│   Jina Reader (Web Pages)        PDF Brochure            │
│   mastersunion.org  ────┐    ┌── brochure.pdf            │
└────────────────────────┐│   │┌──────────────────────────┘
                          ▼▼   ▼▼
                 ┌──────────────────────┐
                 │   Document Chunking   │
                 │  (600 chars, 80 ovlp) │
                 └─────────┬────────────┘
                           ▼
                 ┌──────────────────────┐
                 │   HuggingFace         │
                 │   Embeddings          │
                 │  (all-MiniLM-L6-v2)  │
                 └─────────┬────────────┘
                           ▼
                 ┌──────────────────────┐
                 │    ChromaDB           │
                 │    Vector Store       │
                 └─────────┬────────────┘
                           │
          ┌────────────────▼────────────────┐
          │           RAG CHAIN              │
          │                                  │
    Query ──► Semantic Retrieval (k=8)       │
          │            ▼                     │
          │    Grounded Prompt (EN/HI)       │
          │            ▼                     │
          │   LLaMA 3.3 70B via Groq         │
          │            ▼                     │
          │    Structured Answer             │
          └────────────────┬────────────────┘
                           ▼
                 ┌──────────────────────┐
                 │   Streamlit Chat UI   │
                 │         English      │
                 └──────────────────────┘
```

---

## 📁 Project Structure

```
pgp-ai-advisor/
├── app.py              # Main Streamlit application(api)
├── .env                # API keys (NOT committed to git)
├── app(local).py       # Main Streamlit application (offline)
├── .gitignore          # Ignores .env, chroma_db/, brochure.pdf
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── chroma_db/          # Auto-generated vector store (NOT committed)
```

---

## 🚀 Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/dhruvchhabra29/Team-Paradox-Frosthack.git
cd FROSTHACK
```

### Step 2 — Create virtual environment (Python 3.11 required)

```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

> ⚠️ Python 3.14 is NOT supported — LangChain and Pydantic have compatibility issues with it. Use Python 3.11.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install langchain langchain-community langchain-groq \
            langchain-huggingface langchain-text-splitters \
            chromadb sentence-transformers streamlit \
            pymupdf beautifulsoup4 python-dotenv requests watchdog
```

### Step 4 — Download the program brochure (PDF)

> ⚠️ The brochure PDF is **not included** in this repository. You must download it manually before running the app.

**Option A — via terminal:**
```bash
curl -L -o brochure.pdf "https://files.mastersunion.link/MasterUnion/PGP_Applied_AI_and_Agentic_Systems_Brochure.pdf"
```

**Option B — manually:**
1. Visit: https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems
2. Click **"Download Brochure"**
3. Save the file as `brochure.pdf` in the project root directory

### Step 5 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your API key:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 6 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The first run will:
1. Download the embedding model (~91MB, one-time only)
2. Scrape the program website via Jina Reader
3. Load and parse the PDF brochure
4. Build the ChromaDB vector store

Subsequent runs load from cache and start in seconds.

---

## 🔑 API Key Setup

### Option A — Groq API ✅ Recommended (Free, Fast)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google (takes 2 minutes)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)
5. Paste it into `.env`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

**Free tier:** ~6,000 tokens/minute — sufficient for production demos.
**Model used:** `llama-3.3-70b-versatile`

---

### Option B — Local LLM via Ollama (Offline, No API key needed)

Use this if you have no internet access or want full privacy.

**Step 1 — Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Step 2 — Pull a model:**
```bash
ollama pull llama3.1        # 4.7GB — best quality
# OR
ollama pull mistral         # 4.1GB — faster
# OR
ollama pull phi3:mini       # 2.3GB — lightest (low RAM)
```

**Step 3 — Start Ollama server:**
```bash
ollama serve
```

**Step 4 — Use the `app(local).py`:**

```bash
streamlit run app.py
```

No `.env` file needed — runs 100% locally on your machine.

---

## 🌐 How Web Scraping Works

This app uses **Jina Reader** (`r.jina.ai`) to scrape the Masters' Union website.

Unlike standard scrapers, Jina Reader:
- Handles **JavaScript-rendered pages** (React/Next.js sites)
- Returns **clean plain text** — no HTML noise
- Requires **no API key** and is completely free
- Works with a simple HTTP GET request:

```python
requests.get("https://r.jina.ai/https://mastersunion.org/your-page")
```

Data sources scraped:
| URL | Content |
|-----|---------|
| `/pgp-in-applied-ai-and-agentic-systems` | Program overview, curriculum, fees |
| `/pgp-in-applied-ai-and-agentic-systems-applynow` | Admission details |
| `/about` | Institution background |
| `/placements` | Career outcomes, CTC data |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | LLaMA 3.3 70B (Groq) | Answer generation |
| Embeddings | `all-MiniLM-L6-v2` | Text vectorization |
| Vector Store | ChromaDB | Semantic retrieval |
| Orchestration | LangChain LCEL | RAG pipeline |
| Web Scraping | Jina Reader API | JS-rendered page extraction |
| PDF Parsing | PyMuPDF | Brochure ingestion |
| Frontend | Streamlit | Chat UI |
| Languages | English | support |

---

## 📊 Evaluation Criteria Addressed

| Criterion | Implementation |
|-----------|---------------|
| **Answer Accuracy** | RAG with k=8 retrieval + grounded prompt prevents hallucination |
| **Relevance** | Semantic chunking (600 chars/80 overlap) + multi-source data |
| **Robustness** | Graceful fallback for unknown queries + off-topic redirection |
| **User Experience** | Dark UI, quick question sidebar, bilingual, follow-up suggestions |
| **Technical Implementation** | LCEL pipeline, ChromaDB, HuggingFace embeddings, LLaMA 3.3 70B |

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: langchain.schema` | Use `from langchain_core.documents import Document` |
| `ModuleNotFoundError: langchain.chains` | Use LCEL pipeline instead of `RetrievalQA` |
| `GroqError: api_key not set` | Check `.env` file has `GROQ_API_KEY=gsk_...` and `load_dotenv()` is called |
| `MissingSchema: Invalid URL` | Replace `PASTE_PROGRAM_PAGE_URL_HERE` with actual URL |
| `Python 3.14 pydantic error` | Recreate venv with Python 3.11 |
| Answers say "I don't have info" | Delete `chroma_db/` and rerun to rebuild vector store |
| Duplicate chat input box | Ensure only one `st.chat_input()` with `key="main_input"` |

---

## 📦 requirements.txt

```
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
chromadb
sentence-transformers
streamlit
pymupdf
beautifulsoup4
python-dotenv
requests
watchdog
```

---

## 🔒 .env.example

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🙈 .gitignore

```
.env
chroma_db/
brochure.pdf
__pycache__/
*.pyc
venv/
.DS_Store
*.egg-info/
dist/
```

---

## 👥 Team

**Team Paradox**
* Dhruv Chhabra
* Aayush Ruparelia
* Avaneja Rikhta
* Shivam Kumar

**Newton School of Technology, Rishihood University**
Competing at FrostHack — Xpecto'26, IIT Mandi 🏔️

---

## 📄 License

MIT License — free to use, modify, and build upon.



# frosthack-cloud
