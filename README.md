HR Policy RAG

Project overview
This repository implements a Retrieval-Augmented Generation (RAG) assistant for HR policy documents. It ingests PDF files from the `data/raw` directory, splits them into text chunks, computes embeddings, stores them in a Chroma vector store, and exposes a small FastAPI service that answers user questions using retrieved policy context and a Groq LLM for intent classification and answer generation.

Features
- Ingest PDF files from `data/raw` and split into chunks.
- Compute embeddings using a HuggingFace sentence-transformer and persist vectors in a Chroma database under `chroma_db`.
- Retrieve top-k relevant chunks for a user question.
- Classify intent into `POLICY_QUERY`, `GREETING`, `OUT_OF_SCOPE`, or `UNKNOWN`.
- Generate answers grounded in retrieved policy context using Groq LLM.
- Simple FastAPI HTTP endpoints for health checks, ingestion, intent detection, and querying.

Requirements
- Python 3.10+ recommended.
- See `requirements.txt` for pinned dependencies.

Environment
- Put any required API keys and environment variables in a `.env` file (project uses `python-dotenv`).
- Required environment variable:
	- `GROQ_API_KEY`: API key used by the Groq client.

Default configuration
- Raw documents path: `data/raw`
- Chroma persistence path: `chroma_db`
- Chroma collection name: `hr_policies`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM model: `llama-3.1-8b-instant` (configured via `parameters.py`)

Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and set `GROQ_API_KEY`.
4. Place PDF HR policy documents into `data/raw`.

Ingestion
Endpoint: `POST /ingest`
- This reads all `*.pdf` files in `data/raw`, extracts text per page, splits text into chunks, computes embeddings, and stores them in the Chroma vector store. It returns the number of pages and chunks processed. If no PDFs are found, the endpoint returns a 404 error.

API endpoints
- `POST /ingest` — run ingestion (see Ingestion above).
- `POST /intent` — accepts JSON `{ "question": "..." }`, returns detected intent.
- `POST /results` — accepts JSON `{ "question": "..." }`, returns an answer grounded in retrieved policy context, the detected intent, and a list of source documents/pages.


Files and responsibilities
- `app.py`: FastAPI application and HTTP route handlers.
- `utils.py`: Core logic for loading PDFs, cleaning text, splitting into chunks, embedding, vector store operations, intent detection, and answer generation.
- `parameters.py`: Configuration constants and environment-variable loading.
- `requirements.txt`: Python dependencies.
- `data/raw/`: Place source PDF documents here.
- `chroma_db/`: Directory where Chroma persists vectors and metadata.

Data handling
- PDF text is extracted per page using `pymupdf` and cleaned. Pages with no text are skipped.
- Documents are split using a `RecursiveCharacterTextSplitter` with parameters set in `parameters.py`.
- Each chunk is assigned a deterministic id (SHA-256 of source, page number, and content) before being added to Chroma.

Vector store and models
- Embeddings: `HuggingFaceEmbeddings` using the model configured in `parameters.py`.
- Vector store: Chroma, persisted to `chroma_db`.
- Intent classification and answer generation: Groq LLM (configured via `GROQ_API_KEY` and `GROQ_MODEL` in `parameters.py`).

Running locally
1. Start the API with Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. In a separate terminal run ingestion (optional if you want to pre-populate the vector store):

```bash
curl -X POST http://localhost:8000/ingest
```


