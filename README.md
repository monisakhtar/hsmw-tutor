# 🎓 HSMW Tutor

A **NotebookLM-style Q&A system** for Hochschule Mittweida (HSMW) students. Upload lecture PDFs and ask questions in natural language — the system retrieves relevant passages and generates grounded answers using a local/remote LLM.

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                            ↓
Question → Query Rewriting → Retrieval → LLM Generation → Answer
                  ↑                            ↓
                  └──── Self-Grading (retry if not grounded) ────┘
```

The system uses a **LangGraph agentic pipeline** with 4 nodes:

1. **Query Rewriting** — Rewrites the student's question for better retrieval
2. **Retrieval** — Fetches the top-k most relevant chunks from ChromaDB
3. **Answer Generation** — Generates a grounded answer using the LLM
4. **Self-Grading** — Checks if the answer is grounded in the retrieved docs; retries if not

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| LLM & Embeddings | Ollama (remote HSMW server) |
| Vector Store | ChromaDB |
| RAG Pipeline | LangChain + LangGraph |
| PDF Parsing | PyMuPDF (fitz) |
| Config Management | gin-config |
| Package Manager | uv |

---

## 📁 Project Structure

```
hsmw-tutor/
├── sicim/
│   └── config.gin          # Remote Ollama credentials (not committed to git)
├── chroma_db/              # Persisted vector store
├── app.py                  # Streamlit UI entry point
├── config.py               # App settings & client initialization
├── ingestion.py            # PDF → chunks → ChromaDB pipeline
├── retrieval.py            # ChromaDB retriever
├── chain.py                # LangGraph agentic RAG pipeline
├── pyproject.toml          # uv project config
└── uv.lock                 # Dependency lockfile
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hsmw-tutor.git
cd hsmw-tutor
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.


---

## 📖 Usage

1. **Upload a PDF** — Use the sidebar to upload your lecture notes or slides
2. **Click "Ingest PDF"** — The system will chunk and embed the document
3. **Ask questions** — Type your question in the chat box
4. **Get grounded answers** — The agent retrieves relevant passages and answers based only on your lecture materials

---

## ⚙️ Configuration

All application settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3.2` | LLM model name on Ollama server |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `COLLECTION_NAME` | `hsmw_lectures` | ChromaDB collection name |

Remote server credentials are managed via `sicim/config.gin` (excluded from git).

---

## 🔒 Security

- The `sicim/config.gin` file containing API credentials is **gitignored** and should never be committed
- All data is processed locally — PDFs are not sent to any external service except the HSMW Ollama server for embedding and generation

---

## 📦 Dependencies

```
streamlit
langchain
langchain-community
langchain-ollama
langchain-text-splitters
langgraph
chromadb
pymupdf
gin-config
ollama
```

---

## 🏫 About

Built for **Hochschule Mittweida (HSMW)** as a student Q&A tool for lecture materials.  
A fully local and open source.
