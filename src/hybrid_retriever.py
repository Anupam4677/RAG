"""Hybrid retrieval: dense (OpenAI embeddings via Chroma/HNSW) + sparse (BM25)
combined with an EnsembleRetriever, optionally followed by a cross-encoder
reranking stage for higher top-k precision.
"""
from __future__ import annotations

from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import Field

from src import config
from src.vectorstore import get_all_documents, get_vectorstore

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None and config.USE_RERANKER:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


class RerankingRetriever(BaseRetriever):
    """Wraps a base retriever and reranks its results with a cross-encoder."""

    base_retriever: BaseRetriever = Field(...)
    top_n: int = Field(default=config.RERANK_TOP_N)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []
        reranker = _get_reranker()
        if reranker is None:
            return candidates[: self.top_n]
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in ranked[: self.top_n]:
            doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            results.append(doc)
        return results


def build_hybrid_retriever(top_n: int | None = None) -> BaseRetriever:
    """(Re)builds the hybrid retriever from the current contents of the vector store.

    Call this again after re-ingesting new documents so BM25's in-memory index
    (which has no persistence of its own) picks up the new corpus.
    """
    dense_retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": config.DENSE_TOP_K}
    )

    all_docs = get_all_documents()
    if not all_docs:
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = config.BM25_TOP_K

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[config.HYBRID_DENSE_WEIGHT, config.HYBRID_SPARSE_WEIGHT],
    )

    if config.USE_RERANKER:
        return RerankingRetriever(base_retriever=ensemble, top_n=top_n or config.RERANK_TOP_N)
    return ensemble
