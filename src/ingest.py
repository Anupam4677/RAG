"""End-to-end ingestion pipeline: PDF -> parse -> chunk -> enrich -> embed -> store.

Usage:
    python -m src.ingest                 # ingest every PDF in data_pdf_files/
    python -m src.ingest --files a.pdf b.pdf
    python -m src.ingest --limit 2       # quick smoke test on first N files
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config
from src.metadata import build_documents
from src.pdf_parser import parse_pdf
from src.vectorstore import add_documents, collection_count


def discover_pdfs() -> list[Path]:
    return sorted(config.DATA_DIR.glob("*.pdf"))


def ingest_file(path: Path) -> int:
    print(f"[ingest] parsing {path.name} ...")
    t0 = time.time()
    doc = parse_pdf(str(path))
    print(f"[ingest]   parsed {doc.num_pages} pages in {time.time() - t0:.1f}s")

    docs = build_documents(doc)
    print(f"[ingest]   built {len(docs)} chunks")

    added = add_documents(docs)
    print(f"[ingest]   upserted {added} chunks into Chroma")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest bank report PDFs into the RAG vector store")
    parser.add_argument("--files", nargs="*", help="Specific PDF filenames (in data_pdf_files/) to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N discovered PDFs")
    args = parser.parse_args()

    if args.files:
        pdf_paths = [config.DATA_DIR / f for f in args.files]
        missing = [p for p in pdf_paths if not p.exists()]
        if missing:
            print(f"[ingest] ERROR: files not found: {missing}")
            sys.exit(1)
    else:
        pdf_paths = discover_pdfs()

    if args.limit:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        print(f"[ingest] no PDFs found in {config.DATA_DIR}")
        sys.exit(1)

    print(f"[ingest] found {len(pdf_paths)} PDF(s) to ingest")
    total_chunks = 0
    for path in pdf_paths:
        try:
            total_chunks += ingest_file(path)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] FAILED on {path.name}: {e}")

    print(f"[ingest] done. {total_chunks} chunks upserted this run.")
    print(f"[ingest] collection now holds {collection_count()} total chunks.")


if __name__ == "__main__":
    main()
