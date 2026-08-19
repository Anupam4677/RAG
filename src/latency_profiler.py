"""Component-level latency profiling for the RAG pipeline.

Runs a batch of questions through RAGChain.answer_with_latency, which times
each pipeline stage individually:
  - query_processing_time_ms    : normalizing/transforming the raw question
  - embedding_generation_time_ms: embedding the query into a vector
  - vector_search_latency_ms    : Chroma ANN similarity search
  - bm25_search_latency_ms      : BM25 sparse keyword search
  - reranking_latency_ms        : cross-encoder re-ranking of candidates
  - llm_ttft_ms                 : time to first streamed token from the LLM
  - llm_total_generation_time_ms: total streamed generation time

Results are written to eval/latency_results/<timestamp>.csv and
eval/latency_results/latest.csv for bottleneck analysis.

Usage:
    python -m src.latency_profiler                  # profile the eval-set questions
    python -m src.latency_profiler --n 10            # limit to first 10 questions
    python -m src.latency_profiler --questions "a?" "b?"
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.rag_chain import get_rag_chain


def load_profile_questions() -> list[str]:
    """Reuse the eval dataset's questions (generating a synthetic set if needed)."""
    from src.evaluation import load_eval_set

    return [item["question"] for item in load_eval_set() if item.get("question")]


def run_latency_profile(questions: list[str] | None = None) -> pd.DataFrame:
    chain = get_rag_chain()
    questions = questions or load_profile_questions()

    rows = []
    for question in questions:
        try:
            _, metrics = chain.answer_with_latency(question)
        except Exception as e:  # noqa: BLE001
            print(f"[latency_profiler] failed on {question!r}: {e}")
            continue
        rows.append(metrics.to_row())

    df = pd.DataFrame(rows)
    _save_results(df)
    return df


def _save_results(df: pd.DataFrame) -> None:
    config.LATENCY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(config.LATENCY_RESULTS_DIR / f"{ts}.csv", index=False)
    df.to_csv(config.LATENCY_RESULTS_DIR / "latest.csv", index=False)
    print(f"[latency_profiler] results saved to {config.LATENCY_RESULTS_DIR / f'{ts}.csv'}")


def load_latest_results() -> pd.DataFrame | None:
    path = config.LATENCY_RESULTS_DIR / "latest.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="Limit to the first N questions")
    parser.add_argument("--questions", nargs="+", default=None, help="Explicit list of questions to profile")
    args = parser.parse_args()

    qs = args.questions or load_profile_questions()
    if args.n:
        qs = qs[: args.n]

    result_df = run_latency_profile(qs)
    if len(result_df):
        stage_cols = [c for c in result_df.columns if c.endswith("_ms")]
        print("\nMean latency per stage (ms):")
        print(result_df[stage_cols].mean().to_string())
    else:
        print("[latency_profiler] no results produced")
