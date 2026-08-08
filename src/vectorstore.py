"""Chroma vector store configuration: HNSW index tuned for cosine similarity."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.embeddings import get_embedding_function

_store: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            persist_directory=str(config.CHROMA_DIR),
            collection_metadata={
                "hnsw:space": config.HNSW_SPACE,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:M": config.HNSW_M,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
            },
        )
    return _store


def add_documents(documents: list[Document], batch_size: int = 100) -> int:
    """Add documents in batches, deduped by the deterministic chunk_id metadata."""
    store = get_vectorstore()
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids = [d.metadata["chunk_id"] for d in batch]
        store.add_documents(batch, ids=ids)
        total += len(batch)
    return total


def collection_count() -> int:
    return get_vectorstore()._collection.count()


def get_all_documents() -> list[Document]:
    store = get_vectorstore()
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
