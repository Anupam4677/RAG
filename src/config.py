"""Central configuration for the bank-reports RAG system.

All operational tunables can be overridden via environment variables (or
`.env`) without touching code — see `.env.example` for the full list. Each
falls back to the documented default when unset.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Paths -------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data_pdf_files"
CHROMA_DIR = ROOT_DIR / "chroma_db"
EVAL_DIR = ROOT_DIR / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
EVAL_DATASET_PATH = EVAL_DIR / "eval_dataset.json"
IMAGE_CACHE_DIR = ROOT_DIR / ".image_cache"

for d in (CHROMA_DIR, EVAL_RESULTS_DIR, IMAGE_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Models --------------------------------------------------------------
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = _env_int("EMBEDDING_DIMENSIONS", 3072)  # native dim of text-embedding-3-large
CHAT_MODEL = _env_str("CHAT_MODEL", "gpt-4o-mini")
VISION_MODEL = _env_str("VISION_MODEL", "gpt-4o-mini")

# --- Chunking --------------------------------------------------------------
CHUNK_MIN_TOKENS = _env_int("CHUNK_MIN_TOKENS", 500)
CHUNK_MAX_TOKENS = _env_int("CHUNK_MAX_TOKENS", 1000)
CHUNK_OVERLAP_RATIO = _env_float("CHUNK_OVERLAP_RATIO", 0.15)  # 15%, within the 10-20% target band
TOKENIZER_ENCODING = _env_str("TOKENIZER_ENCODING", "cl100k_base")

# --- Vector store ------------------------------------------------------
COLLECTION_NAME = _env_str("COLLECTION_NAME", "bank_annual_reports")
HNSW_SPACE = _env_str("HNSW_SPACE", "cosine")
HNSW_CONSTRUCTION_EF = _env_int("HNSW_CONSTRUCTION_EF", 200)
HNSW_M = _env_int("HNSW_M", 16)
HNSW_SEARCH_EF = _env_int("HNSW_SEARCH_EF", 100)

# --- Retrieval -----------------------------------------------------------
DENSE_TOP_K = _env_int("DENSE_TOP_K", 10)
BM25_TOP_K = _env_int("BM25_TOP_K", 10)
HYBRID_DENSE_WEIGHT = _env_float("HYBRID_DENSE_WEIGHT", 0.6)
HYBRID_SPARSE_WEIGHT = _env_float("HYBRID_SPARSE_WEIGHT", 0.4)
RERANK_TOP_N = _env_int("RERANK_TOP_N", 5)
RERANKER_MODEL = _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = _env_bool("USE_RERANKER", True)

# --- Multimodal (image captioning) --------------------------------------
CAPTION_IMAGES = _env_bool("CAPTION_IMAGES", True)
MIN_IMAGE_SIZE_PX = _env_int("MIN_IMAGE_SIZE_PX", 80)  # skip tiny decorative icons/logos
MAX_IMAGES_PER_PAGE = _env_int("MAX_IMAGES_PER_PAGE", 4)

# --- Noise removal ---------------------------------------------------------
HEADER_FOOTER_MARGIN_RATIO = _env_float("HEADER_FOOTER_MARGIN_RATIO", 0.08)  # top/bottom 8% of page treated as header/footer band
MIN_REPEAT_RATIO_FOR_BOILERPLATE = _env_float("MIN_REPEAT_RATIO_FOR_BOILERPLATE", 0.4)  # a line seen on >=40% of pages is boilerplate
