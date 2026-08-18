from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import (
    ingest_documents,
    retrieve_documents,
    generate_answer,
    detect_intent,
)


app = FastAPI(
    title="HR Policy RAG",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str




""" Runs only at the time of updating the policies. 
"""

@app.post("/ingest")
def ingest():
    result = ingest_documents()

    if result["pages"] == 0:
        raise HTTPException(
            status_code=404,
            detail="No PDF files found in data/raw",
        )

    return {
        "message": "Ingestion completed",
        **result,
    }

@app.post("/intent")
def intent(request: QueryRequest):
    detected_intent = detect_intent(
        request.question
    )

    return {
        "question": request.question,
        "intent": detected_intent,
    }

@app.post("/results")
async def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    intent = detect_intent(question)

    if intent == "GREETING":
        return {
            "answer": "Hello! How can I help you with the HR policies?",
            "intent": intent,
            "sources": [],
        }

    if intent == "OUT_OF_SCOPE":
        return {
            "answer": (
                "I can only help with questions "
                "related to the provided HR policies."
            ),
            "intent": intent,
            "sources": [],
        }

    if intent == "UNKNOWN":
        return {
            "answer": (
                "I'm not sure what you're asking. "
                "Could you rephrase your question?"
            ),
            "intent": intent,
            "sources": [],
        }

    # POLICY_QUERY
    documents = retrieve_documents(question)

    if not documents:
        return {
            "answer": (
                "I could not find this information "
                "in the provided HR policy."
            ),
            "intent": intent,
            "sources": [],
        }

    answer = generate_answer(
        question,
        documents,
    )

    sources = [
        {
            "file_name": document.metadata.get(
                "file_name"
            ),
            "page_number": document.metadata.get(
                "page_number"
            ),
        }
        for document in documents
    ]

    return {
        "answer": answer,
        "intent": intent,
        "sources": sources,
    }