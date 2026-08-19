# Bank Annual Reports RAG System

A production-lean Retrieval-Augmented Generation system for querying bank annual
reports, investor presentations, and quarterly results (PDFs). Built with
Python, LangChain, OpenAI embeddings/chat models, ChromaDB, and Gradio.

## Architecture

```
data_pdf_files/*.pdf
        │
        ▼
┌───────────────────┐   PyMuPDF (text + font-size heading detection)
│   src/pdf_parser   │   pdfplumber (tables → markdown)
│   parse + clean    │   GPT-4o-mini vision (image/chart captions)
└─────────┬──────────┘   boilerplate stripping (headers/footers/page numbers)
          ▼
┌───────────────────┐   Split page markdown into sections by header
│   src/chunker      │   Semantic chunking (LangChain SemanticChunker) per section
│  standardized      │   Rebalanced to 500-1000 token band, ~15% overlap
│  re-chunking       │   Tables/image captions kept as atomic chunks
└─────────┬──────────┘
          ▼
┌───────────────────┐   source, page, section, content_type, doc_summary,
│   src/metadata     │   chunk_id (deterministic), token_count, ingested_at,
│   enrichment       │   access_level
└─────────┬──────────┘
          ▼
┌───────────────────┐   OpenAI text-embedding-3-large, L2-normalized
│ src/embeddings +   │   ChromaDB, HNSW index (cosine space)
│ src/vectorstore    │
└─────────┬──────────┘
          ▼
┌───────────────────┐   Dense (Chroma/HNSW) + Sparse (BM25) via EnsembleRetriever
│ src/hybrid_        │   Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│ retriever          │
└─────────┬──────────┘
          ▼
┌───────────────────┐   Grounded answer generation (gpt-4o-mini) with
│  src/rag_chain     │   mandatory source+page citations
└─────────┬──────────┘
          ▼
      app.py (Gradio UI: Chat / Ingest / Evaluation tabs)

src/evaluation.py — Hit Rate, Precision/Recall@k, MRR, NDCG, mean embedding
                    similarity, scored against a labeled or auto-generated
                    (synthetic) eval set, results stored under eval/results/
```

## Design decisions & rationale

| Concern | Choice | Why |
|---|---|---|
| Parsing | Custom PyMuPDF + pdfplumber pipeline | Avoids heavy system deps (poppler/tesseract) that `unstructured`/LlamaParse typically require on Windows, while still recovering headings (font-size heuristics), tables (pdfplumber), and images (PyMuPDF image extraction). |
| Headers/footers/watermarks | Frequency-based boilerplate detection | Lines repeated across ≥40% of a document's pages (after normalizing digits) are stripped; bare page-number lines are regex-filtered. |
| Multimodal | GPT-4o-mini vision captions per extracted image/chart | Bank investor decks are chart-heavy; captions make chart content full-text-searchable. Captions are cached by image hash so re-ingestion doesn't re-pay the API cost. |
| Chunking | Section-aware semantic chunking, rebalanced to 500-1000 tokens, ~15% overlap | Splits on meaning (embedding-distance breakpoints) rather than fixed character counts, then a deterministic token splitter guarantees every chunk still lands in the target size band. Tables are never split mid-row. |
| Metadata | Source, page, section, content_type, chunk_id, token_count, doc_summary, ingested_at, access_level on every chunk | Chunk_id is a deterministic hash of (file, page, type, content) so re-ingestion **upserts** instead of duplicating. `doc_summary` (LLM-generated, cached) is injected into every child chunk so a narrow chunk retrieved alone still carries global document context ("parent-document summary in child chunk"). `access_level` is a placeholder hook for multi-tenant ACL filtering. |
| Embeddings | `text-embedding-3-large`, L2-normalized, fixed dimension | Pinned model + dimension because mixing embedding models/dimensions mid-index silently corrupts similarity search. Normalizing makes cosine and dot-product equivalent regardless of store config. |
| Vector store | ChromaDB, HNSW, `hnsw:space=cosine` | Approximate nearest neighbor at scale; cosine matches the normalized embeddings. |
| Retrieval | Hybrid: dense (HNSW) + sparse (BM25) via `EnsembleRetriever`, then cross-encoder rerank | Dense search catches semantic matches; BM25 catches exact keyword/number matches (critical for financial figures like "₹1,234 crore" or "Q1FY27") that embeddings can blur. The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the merged candidate set for top-k precision. |
| Generation | `gpt-4o-mini`, temperature 0, citation-mandatory system prompt | Cheap and fast; the prompt forces citing (source, page) for every claim and forces the model to say "not in context" instead of hallucinating — important for financial data. |
| Evaluation | Retrieval metrics (Hit Rate, P/R@k, MRR, NDCG) + mean query-chunk cosine similarity | Retrieval metrics need labeled (question → source+page) pairs; if none exist yet, a synthetic set is bootstrapped by sampling indexed chunks and asking the LLM to write the question each one answers — replace with real analyst questions over time for a trustworthy eval set. |

## Project layout

```
RAG/
├── .env                      # API keys + operational config overrides (see .env.example)
├── .env.example               # documents every supported override, with defaults
├── requirements.txt
├── data_pdf_files/            # source PDFs (bank annual reports etc.)
├── chroma_db/                  # persisted vector store (created at runtime)
├── .image_cache/                # cached image→caption mapping (created at runtime)
├── eval/
│   ├── eval_dataset.json        # labeled or auto-generated Q/A eval set
│   └── results/                  # timestamped evaluation runs (JSON + CSV)
├── src/
│   ├── config.py                 # all tunables in one place
│   ├── pdf_parser.py              # parse + clean PDFs
│   ├── chunker.py                  # semantic re-chunking
│   ├── metadata.py                  # metadata enrichment
│   ├── embeddings.py                 # OpenAI embeddings wrapper (normalized)
│   ├── vectorstore.py                 # Chroma/HNSW config
│   ├── hybrid_retriever.py             # dense+BM25 ensemble + reranker
│   ├── ingest.py                        # ingestion pipeline / CLI
│   ├── rag_chain.py                      # retrieval + grounded generation
│   └── evaluation.py                      # retrieval/embedding metrics
├── app.py                        # Gradio UI (Chat / Ingest / Evaluation)
└── USER_MANUAL.md
```

See **USER_MANUAL.md** for setup and day-to-day usage instructions.

## Extending this system

- **Real eval labels**: replace `eval/eval_dataset.json` with analyst-written
  questions and their true (source file, page) once available — synthetic
  labels are a bootstrap, not a substitute for human-labeled ground truth.
- **Multi-tenancy / access control**: `access_level` metadata is already on
  every chunk; add a `filter={"access_level": ...}` to retriever calls to
  enforce it.
- **Larger corpora**: Chroma's local HNSW index scales to the low millions of
  vectors on one machine; beyond that, consider a managed vector DB (Pinecone,
  Weaviate, Qdrant) — the `vectorstore.py` abstraction is the only place that
  would need to change.
