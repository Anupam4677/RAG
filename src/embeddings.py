"""OpenAI embeddings wrapper.

Wraps langchain_openai.OpenAIEmbeddings and adds L2 normalization so vector
magnitude never distorts similarity, keeping us robust regardless of the
vector store's distance metric (cosine or dot-product-on-normalized-vectors
are equivalent). Model + dimension are pinned in config: do not change either
without a full re-index, or old and new vectors will not be comparable.
"""
from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src import config


class NormalizedOpenAIEmbeddings(Embeddings):
    def __init__(self) -> None:
        self._impl = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
            dimensions=config.EMBEDDING_DIMENSIONS,
        )

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> list[list[float]]:
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._normalize(self._impl.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        vec = self._impl.embed_query(text)
        return self._normalize([vec])[0]


def get_embedding_function() -> Embeddings:
    return NormalizedOpenAIEmbeddings()
