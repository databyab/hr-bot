import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

RAW_DATA_PATH = "data/raw"
CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "hr_policies"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 4

GROQ_MODEL = "llama-3.1-8b-instant"