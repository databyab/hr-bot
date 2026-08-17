from pathlib import Path
import hashlib
import re
import time
import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from parameters import (
    RAW_DATA_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL,
)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdfs() -> list[Document]:
    documents = []

    for pdf_file in Path(RAW_DATA_PATH).glob("*.pdf"):
        pdf = pymupdf.open(str(pdf_file))

        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(
                page.get_text("text")
            )

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_file),
                        "file_name": pdf_file.name,
                        "page_number": page_number,
                    },
                )
            )

        pdf.close()

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


def get_document_id(document: Document) -> str:
    content = (
        document.metadata["source"]
        + str(document.metadata["page_number"])
        + document.page_content
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def ingest_documents():
    documents = load_pdfs()

    if not documents:
        return {
            "pages": 0,
            "chunks": 0,
        }

    chunks = split_documents(documents)

    vectorstore = get_vectorstore()

    ids = [
        get_document_id(chunk)
        for chunk in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids,
    )

    return {
        "pages": len(documents),
        "chunks": len(chunks),
    }


def retrieve_documents(question: str) -> list[Document]:
    start = time.perf_counter()

    vectorstore = get_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=TOP_K,
    )

    print(
        f"Retrieval time: "
        f"{time.perf_counter() - start:.3f}s"
    )

    return documents

def detect_intent(question: str) -> str:
    prompt = f"""
Classify the user's message into exactly one of these intents:

POLICY_QUERY
GREETING
OUT_OF_SCOPE
UNKNOWN

Definitions:

GREETING:
Only classify as GREETING when the user is actually greeting the assistant.
Examples:
- Hi
- Hello
- Hey
- Good morning

Do NOT classify questions as GREETING, even if they contain words
that could appear in casual conversation.

POLICY_QUERY:
A question asking for information that could be answered
from the provided HR policy documents.

OUT_OF_SCOPE:
A meaningful question that is not related to the HR policies.

UNKNOWN:
The input is ambiguous, nonsensical, or impossible to classify.

User message:
{question}

Return ONLY the intent name.
"""

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)

    intent = response.content.strip().upper()

    valid_intents = {
        "POLICY_QUERY",
        "GREETING",
        "OUT_OF_SCOPE",
        "UNKNOWN",
    }

    if intent not in valid_intents:
        return "UNKNOWN"

    return intent

def generate_answer(
    question: str,
    documents: list[Document],
) -> str:

    start = time.perf_counter()

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are an HR policy assistant.

Answer the question using only the provided
HR policy context.

Do not invent or assume information.

If the answer is not present in the context,
say that the information is not available in
the provided HR policy.

HR POLICY CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)
    
    return response.content