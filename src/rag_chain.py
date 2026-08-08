"""RAG question-answering chain: hybrid retrieval -> grounded generation with citations."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src import config
from src.hybrid_retriever import build_hybrid_retriever

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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_turn, ai_turn in history or []:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": ai_turn})
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            }
        )

        response = self.llm.invoke(messages)
        sources = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page_num"),
                "section": d.metadata.get("section_title"),
                "content_type": d.metadata.get("content_type"),
                "snippet": d.page_content[:220],
            }
            for d in docs
        ]
        return RAGAnswer(answer=response.content, sources=sources)


_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
