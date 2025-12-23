from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="RAG API",
    description="FastAPI + LangChain RAG application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ----------- Request Schema -----------

class Query(BaseModel):
    question: str


# ----------- Routes -----------

@app.get("/")
def root():
    return {
        "message": "RAG API is running 🚀",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/ask")
def ask(query: Query):
    """
    Ask a question to the RAG pipeline.
    """
    try:
        # Lazy import to avoid OpenAPI/schema issues
        from app.rag import get_rag_answer

        answer = get_rag_answer(query.question)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
