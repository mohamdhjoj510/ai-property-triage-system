"""Populate the RAG service's ChromaDB collection from synthetic listings.

Reads `data/synthetic-listings/listings.json`, embeds each listing with a
sentence-transformers model, and stores them in a persistent ChromaDB
collection at `services/rag-service/chroma_db`.

Re-running this script is safe: it upserts by listing id, so existing
records are updated rather than duplicated.
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

SCRIPT_DIR = Path(__file__).resolve().parent
LISTINGS_PATH = SCRIPT_DIR.parent.parent / "data" / "synthetic-listings" / "listings.json"
CHROMA_PATH = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "property_listings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_listings(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_document(listing: dict) -> str:
    """Flatten a listing into a single text document for embedding."""
    features = ", ".join(listing.get("features", []))
    return (
        f"Title: {listing['title']}\n"
        f"Type: {listing['property_type']}\n"
        f"Location: {listing['location']}\n"
        f"Price: {listing['price']}\n"
        f"Rooms: {listing['rooms']}\n"
        f"Features: {features}\n"
        f"Condition: {listing['condition']}\n"
        f"Description: {listing['description']}"
    )


def build_metadata(listing: dict) -> dict:
    return {
        "id": listing["id"],
        "property_type": listing["property_type"],
        "location": listing["location"],
        "price": listing["price"],
        "rooms": listing["rooms"],
        "condition": listing["condition"],
    }


def main() -> None:
    listings = load_listings(LISTINGS_PATH)
    print(f"Loaded {len(listings)} listings from {LISTINGS_PATH}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    ids = [str(listing["id"]) for listing in listings]
    documents = [build_document(listing) for listing in listings]
    metadatas = [build_metadata(listing) for listing in listings]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Inserted {len(ids)} documents into collection '{COLLECTION_NAME}'")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
