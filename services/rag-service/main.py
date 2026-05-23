"""RAG service — semantic similarity search over the property listings corpus.

Uses the ChromaDB collection populated by `populate_chroma.py` and the same
sentence-transformers embedding model so query vectors are comparable to the
stored document vectors. When the query mentions a specific property type
(office, retail, etc.), the search is metadata-filtered to that type and
falls back to an unfiltered semantic search if the filter eliminates
everything. LLM-based answer generation is not wired in yet — the insight
is a deterministic sentence built from the top result.
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "property_listings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Order matters: commercial-specific keywords first, then villa (more specific
# than house), then house, then apartment. Each entry is (property_type, keywords).
# Word-boundary matching avoids false positives like "warehouse" → "house".
TYPE_DETECTION = (
    ("office", ("office",)),
    ("retail", ("retail", "shop", "storefront")),
    ("industrial", ("industrial", "warehouse")),
    ("villa", ("villa",)),
    ("house", ("house",)),
    ("apartment", ("apartment",)),
)

PROPERTY_TYPE_PLURALS = {
    "apartment": "apartments",
    "house": "houses",
    "villa": "villas",
    "office": "offices",
    "retail": "retail spaces",
    "industrial": "industrial properties",
}

app = FastAPI(title="rag-service")

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,
)


class QueryRequest(BaseModel):
    description: str


def detect_property_type(text: str) -> str | None:
    """Return a property_type if the text clearly mentions one, else None."""
    lowered = text.lower()
    for property_type, keywords in TYPE_DETECTION:
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                return property_type
    return None


def _run_query(description: str, property_type: str | None) -> dict:
    kwargs: dict = {"query_texts": [description], "n_results": TOP_K}
    if property_type:
        kwargs["where"] = {"property_type": property_type}
    return _collection.query(**kwargs)


def _pluralize_property_type(property_type: str) -> str:
    return PROPERTY_TYPE_PLURALS.get(property_type, f"{property_type}s")


def _city_from_location(location: str) -> str:
    return location.split(",")[0].strip()


def build_similar_listings(query_result: dict) -> list[dict]:
    """Flatten ChromaDB's column-major query result into a list of listing dicts."""
    metadatas = (query_result.get("metadatas") or [[]])[0]
    documents = (query_result.get("documents") or [[]])[0]
    similar = []
    for meta, doc in zip(metadatas, documents):
        similar.append(
            {
                "id": meta.get("id"),
                "property_type": meta.get("property_type"),
                "location": meta.get("location"),
                "price": meta.get("price"),
                "condition": meta.get("condition"),
                "document": doc,
            }
        )
    return similar


def build_insight(top_listing: dict) -> str:
    if not top_listing:
        return "No similar listings found in the corpus yet."
    plural = _pluralize_property_type(top_listing.get("property_type", "listing"))
    city = _city_from_location(top_listing.get("location", "your area"))
    condition = top_listing.get("condition", "")
    if condition == "needs renovation":
        return f"This listing is similar to {plural} in {city} that need renovation."
    if condition:
        return f"This listing is similar to {condition} {plural} in {city}."
    return f"This listing is similar to {plural} in {city}."


@app.get("/")
def root():
    return {"service": "rag-service", "status": "running"}


@app.post("/query")
def query(request: QueryRequest):
    detected_type = detect_property_type(request.description)
    result = _run_query(request.description, detected_type)

    ids = (result.get("ids") or [[]])[0]
    if detected_type and not ids:
        # Filter eliminated everything — fall back to unfiltered semantic search.
        result = _run_query(request.description, None)

    similar = build_similar_listings(result)
    insight = build_insight(similar[0] if similar else {})
    return {
        "similar_listings": similar,
        "insight": insight,
    }
