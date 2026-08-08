"""Document re-chunking: turns parsed PDF pages into standardized-length chunks.

Strategy:
  - Body markdown text is split *semantically* (embedding-distance breakpoints
    via LangChain's SemanticChunker) and then any resulting piece outside the
    500-1000 token band is re-split/merged with a token-aware recursive
    splitter so every chunk still respects the production-friendly size band
    with ~15% overlap.
  - Tables and image captions are kept as atomic chunks (never split) since
    breaking a table mid-row destroys its meaning; oversized tables are
    row-chunked instead of token-chunked so every piece stays a valid table.
"""
from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.pdf_parser import DocumentContent, PageContent

_encoding = tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


_semantic_chunker: SemanticChunker | None = None


def _get_semantic_chunker() -> SemanticChunker:
    global _semantic_chunker
    if _semantic_chunker is None:
        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
        _semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
    return _semantic_chunker


_token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=config.TOKENIZER_ENCODING,
    chunk_size=config.CHUNK_MAX_TOKENS,
    chunk_overlap=int(config.CHUNK_MAX_TOKENS * config.CHUNK_OVERLAP_RATIO),
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _rebalance_to_token_band(pieces: list[str]) -> list[str]:
    """Merge small adjacent pieces / split oversized pieces to hit the target band."""
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}\n\n{piece}" if buffer else piece
        if count_tokens(candidate) <= config.CHUNK_MAX_TOKENS:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)

    final: list[str] = []
    for piece in merged:
        if count_tokens(piece) > config.CHUNK_MAX_TOKENS:
            final.extend(_token_splitter.split_text(piece))
        else:
            final.append(piece)
    return final


def _semantic_split(text: str) -> list[str]:
    if count_tokens(text) <= config.CHUNK_MAX_TOKENS:
        return [text] if text.strip() else []
    try:
        pieces = _get_semantic_chunker().split_text(text)
    except Exception as e:  # noqa: BLE001 - fall back to deterministic splitter
        print(f"[chunker] semantic chunking failed ({e}); falling back to token splitter")
        return _token_splitter.split_text(text)
    return _rebalance_to_token_band(pieces)


def _chunk_table_rows(markdown_table: str) -> list[str]:
    """Row-chunk an oversized markdown table, repeating the header in each piece."""
    lines = markdown_table.split("\n")
    if len(lines) <= 2:
        return [markdown_table]
    header, sep, *rows = lines
    prefix = f"{header}\n{sep}\n"
    chunks, buffer = [], prefix
    for row in rows:
        candidate = buffer + row + "\n"
        if count_tokens(candidate) > config.CHUNK_MAX_TOKENS and buffer != prefix:
            chunks.append(buffer.rstrip("\n"))
            buffer = prefix + row + "\n"
        else:
            buffer = candidate
    if buffer != prefix:
        chunks.append(buffer.rstrip("\n"))
    return chunks or [markdown_table]


def _split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split page markdown into (section_title, section_body) using '#' headers.

    Text before the first header is kept under section_title="" so nothing is lost.
    """
    lines = markdown_text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("#"):
            if current_body:
                sections.append((current_title, current_body))
            current_title = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, current_body))
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def chunk_page(page: PageContent) -> list[dict]:
    """Returns a list of {"text", "content_type", "section_title"} dicts for one page."""
    results: list[dict] = []

    for section_title, section_text in _split_into_sections(page.markdown_text):
        for piece in _semantic_split(section_text):
            if piece.strip():
                results.append(
                    {"text": piece, "content_type": "text", "section_title": section_title}
                )

    for table in page.tables:
        if count_tokens(table.markdown) > config.CHUNK_MAX_TOKENS:
            for row_chunk in _chunk_table_rows(table.markdown):
                results.append({"text": row_chunk, "content_type": "table", "section_title": ""})
        else:
            results.append({"text": table.markdown, "content_type": "table", "section_title": ""})

    for caption in page.image_captions:
        results.append({"text": caption, "content_type": "image_caption", "section_title": ""})

    return results


def chunk_document(doc: DocumentContent) -> list[dict]:
    """Returns a flat list of {"text", "content_type", "page_num"} dicts for the doc."""
    all_chunks = []
    for page in doc.pages:
        for chunk in chunk_page(page):
            chunk["page_num"] = page.page_num
            all_chunks.append(chunk)
    return all_chunks
