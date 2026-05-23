"""RAG service — minimal FastAPI skeleton.

Returns dummy responses for now. Real retrieval and generation will be added later.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="rag-service")


class QueryRequest(BaseModel):
    description: str


@app.get("/")
def root() -> dict:
    return {"service": "rag-service", "status": "running"}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return {
        "similar_listings": [
            {"id": 1, "summary": "Modern apartment in Haifa"},
            {"id": 2, "summary": "Renovated beach apartment"},
        ],
        "insight": "This property is similar to renovated apartments in northern Haifa.",
    }
