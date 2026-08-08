"""Central configuration for the bank-reports RAG system."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

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
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # native dim of text-embedding-3-large
CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

# --- Chunking --------------------------------------------------------------
CHUNK_MIN_TOKENS = 500
CHUNK_MAX_TOKENS = 1000
CHUNK_OVERLAP_RATIO = 0.15  # 15%, within the 10-20% target band
TOKENIZER_ENCODING = "cl100k_base"

# --- Vector store ------------------------------------------------------
COLLECTION_NAME = "bank_annual_reports"
HNSW_SPACE = "cosine"
HNSW_CONSTRUCTION_EF = 200
HNSW_M = 16
HNSW_SEARCH_EF = 100

# --- Retrieval -----------------------------------------------------------
DENSE_TOP_K = 10
BM25_TOP_K = 10
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4
RERANK_TOP_N = 5
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

# --- Multimodal (image captioning) --------------------------------------
CAPTION_IMAGES = True
MIN_IMAGE_SIZE_PX = 80  # skip tiny decorative icons/logos
MAX_IMAGES_PER_PAGE = 4

# --- Noise removal ---------------------------------------------------------
HEADER_FOOTER_MARGIN_RATIO = 0.08  # top/bottom 8% of page treated as header/footer band
MIN_REPEAT_RATIO_FOR_BOILERPLATE = 0.4  # a line seen on >=40% of pages is boilerplate
