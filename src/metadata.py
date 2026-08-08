"""Metadata enrichment: turns raw chunk dicts into LangChain Documents with
production-grade metadata attached to every single chunk.

Each chunk gets:
  - source (filename), source_path, page_num, section_title
  - content_type (text | table | image_caption)
  - chunk_id (stable, deterministic -> safe for idempotent re-ingestion)
  - token_count
  - doc_title, doc_summary (parent-document summary injected into every child
    chunk so retrieval keeps global context even from a narrow chunk)
  - ingested_at (ISO timestamp)
  - access_level (defaults to "internal"; placeholder hook for multi-tenancy)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src import config
from src.chunker import chunk_document, count_tokens
from src.pdf_parser import DocumentContent

_summary_cache: dict[str, str] = {}


def _summarize_document(doc: DocumentContent) -> str:
    """One-line-per-document summary, generated once and cached for re-ingestion."""
    if doc.source_path in _summary_cache:
        return _summary_cache[doc.source_path]

    sample_pages = doc.pages[:3] + (doc.pages[-1:] if len(doc.pages) > 3 else [])
    sample_text = "\n\n".join(p.markdown_text for p in sample_pages)[:8000]
    if not sample_text.strip():
        return ""

    try:
        llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)
        prompt = (
            "Summarize this bank report/document in 2-3 sentences: name the "
            "institution, the document type (e.g. annual report, investor "
            f"presentation, quarterly results), and reporting period if visible.\n\n"
            f"Filename: {doc.filename}\n\nExcerpt:\n{sample_text}"
        )
        summary = llm.invoke(prompt).content.strip()
    except Exception as e:  # noqa: BLE001 - summary is a nice-to-have, never block ingestion
        print(f"[metadata] doc summary failed for {doc.filename}: {e}")
        summary = f"Document: {doc.filename}"

    _summary_cache[doc.source_path] = summary
    return summary


def _chunk_id(source_path: str, page_num: int, content_type: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_path}|p{page_num}|{content_type}|{digest}"


def build_documents(doc: DocumentContent, access_level: str = "internal") -> list[Document]:
    doc_summary = _summarize_document(doc)
    ingested_at = datetime.now(timezone.utc).isoformat()
    raw_chunks = chunk_document(doc)

    documents: list[Document] = []
    seen_chunk_ids: set[str] = set()
    for i, chunk in enumerate(raw_chunks):
        text = chunk["text"]
        chunk_id = _chunk_id(doc.source_path, chunk["page_num"], chunk["content_type"], text)
        if chunk_id in seen_chunk_ids:
            # Identical (page, content_type, text) already emitted - e.g. a table
            # pdfplumber double-detected, or a repeated line the boilerplate filter
            # missed. Indexing the same text twice adds no retrieval value and
            # would otherwise crash the Chroma upsert (duplicate IDs in one batch).
            continue
        seen_chunk_ids.add(chunk_id)

        metadata = {
            "source": doc.filename,
            "source_path": doc.source_path,
            "doc_title": doc.title,
            "doc_summary": doc_summary,
            "page_num": chunk["page_num"],
            "total_pages": doc.num_pages,
            "section_title": chunk.get("section_title", ""),
            "content_type": chunk["content_type"],
            "chunk_id": chunk_id,
            "chunk_index_in_doc": i,
            "token_count": count_tokens(text),
            "ingested_at": ingested_at,
            "access_level": access_level,
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents
