"""RAG question-answering chain: hybrid retrieval -> grounded generation with citations."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src import config
from src.embeddings import get_embedding_function
from src.hybrid_retriever import RerankingRetriever, _get_reranker, build_hybrid_retriever
from src.vectorstore import get_vectorstore

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about \
bank annual reports, investor presentations, and quarterly results.

Answer ONLY using the provided context chunks. Each chunk is labeled with its \
source document and page number. Rules:
- If the answer isn't in the context, say so plainly - never guess or use outside knowledge.
- Always cite the source and page number for any figure or claim, using the actual \
filename and page number shown in the context, e.g.: (Source: q1fy27-earnings-presentation.pdf, p.12).
- When quoting financial figures, reproduce them exactly as they appear (numbers, units, %, currency).
- If context chunks conflict (e.g. different quarters), point out the discrepancy instead of silently picking one.
- Be concise and structured; use bullet points for multi-figure answers."""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class LatencyMetrics:
    """Wall-clock time (ms) spent in each stage of a single RAGChain.answer_with_latency call."""

    question: str
    query_processing_time_ms: float
    embedding_generation_time_ms: float
    vector_search_latency_ms: float
    bm25_search_latency_ms: float
    reranking_latency_ms: float
    llm_ttft_ms: float
    llm_total_generation_time_ms: float
    total_latency_ms: float
    num_docs_retrieved: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return asdict(self)


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = (
            f"[{i}] Source: {meta.get('source')} | Page: {meta.get('page_num')} | "
            f"Section: {meta.get('section_title') or 'N/A'} | Type: {meta.get('content_type')}"
        )
        doc_summary = meta.get("doc_summary", "")
        blocks.append(f"{header}\nDocument context: {doc_summary}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    messages = []
    for user_turn, ai_turn in history:
        messages.append(HumanMessage(content=user_turn))
        messages.append(AIMessage(content=ai_turn))
    return messages


def _build_messages(question: str, context: str, history: list[tuple[str, str]] | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_turn, ai_turn in history or []:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": ai_turn})
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"})
    return messages


def _build_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page_num"),
            "section": d.metadata.get("section_title"),
            "content_type": d.metadata.get("content_type"),
            "snippet": d.page_content[:220],
        }
        for d in docs
    ]


class RAGChain:
    def __init__(self) -> None:
        self.retriever = build_hybrid_retriever()
        self.llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)

    def refresh_retriever(self) -> None:
        """Call after re-ingestion so BM25's index picks up new documents."""
        self.retriever = build_hybrid_retriever()

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> RAGAnswer:
        docs = self.retriever.invoke(question)
        context = _format_context(docs)
        messages = _build_messages(question, context, history)
        response = self.llm.invoke(messages)
        return RAGAnswer(answer=response.content, sources=_build_sources(docs))

    def answer_with_latency(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> tuple[RAGAnswer, LatencyMetrics]:
        """Same as answer(), but times each pipeline stage individually.

        Retrieval is re-run stage-by-stage (embed -> dense ANN search -> BM25 ->
        rerank) rather than through the composed retriever, so each stage's cost is
        isolated. This duplicates some work (e.g. the composed retriever would
        embed once internally) which is fine for a diagnostic/profiling call but
        means it should not replace answer() on the hot chat path.
        """
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        normalized_question = question.strip()
        query_processing_time_ms = (time.perf_counter() - t0) * 1000

        embed_fn = get_embedding_function()
        t0 = time.perf_counter()
        query_vector = embed_fn.embed_query(normalized_question)
        embedding_generation_time_ms = (time.perf_counter() - t0) * 1000

        store = get_vectorstore()
        t0 = time.perf_counter()
        dense_docs = store.similarity_search_by_vector(query_vector, k=config.DENSE_TOP_K)
        vector_search_latency_ms = (time.perf_counter() - t0) * 1000

        base = self.retriever.base_retriever if isinstance(self.retriever, RerankingRetriever) else self.retriever
        bm25_docs: list[Document] = []
        bm25_search_latency_ms = 0.0
        if isinstance(base, EnsembleRetriever) and len(base.retrievers) > 1:
            t0 = time.perf_counter()
            bm25_docs = base.retrievers[1].invoke(normalized_question)
            bm25_search_latency_ms = (time.perf_counter() - t0) * 1000

        seen = {d.page_content for d in dense_docs}
        candidates = dense_docs + [d for d in bm25_docs if d.page_content not in seen]

        reranking_latency_ms = 0.0
        if config.USE_RERANKER and candidates:
            reranker = _get_reranker()
            if reranker is not None:
                t0 = time.perf_counter()
                pairs = [(normalized_question, doc.page_content) for doc in candidates]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                docs = [doc for doc, _ in ranked[: config.RERANK_TOP_N]]
                reranking_latency_ms = (time.perf_counter() - t0) * 1000
            else:
                docs = candidates[: config.RERANK_TOP_N]
        else:
            docs = candidates

        context = _format_context(docs)
        messages = _build_messages(question, context, history)

        t0 = time.perf_counter()
        first_token_time = None
        chunks = []
        for chunk in self.llm.stream(messages):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(chunk.content or "")
        t_gen_end = time.perf_counter()
        llm_ttft_ms = ((first_token_time or t_gen_end) - t0) * 1000
        llm_total_generation_time_ms = (t_gen_end - t0) * 1000

        metrics = LatencyMetrics(
            question=question,
            query_processing_time_ms=query_processing_time_ms,
            embedding_generation_time_ms=embedding_generation_time_ms,
            vector_search_latency_ms=vector_search_latency_ms,
            bm25_search_latency_ms=bm25_search_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
            llm_ttft_ms=llm_ttft_ms,
            llm_total_generation_time_ms=llm_total_generation_time_ms,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
            num_docs_retrieved=len(docs),
        )
        return RAGAnswer(answer="".join(chunks), sources=_build_sources(docs)), metrics


_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
