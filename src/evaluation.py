"""Retrieval / embedding quality evaluation.

Metrics computed against a labeled eval set (question -> expected source file
+ page, i.e. "which chunk should retrieval have found"):
  - Hit Rate@k          : fraction of questions where a relevant chunk is in top-k
  - Precision@k         : relevant chunks / k, averaged
  - Recall@k            : relevant chunks found / total relevant, averaged
  - MRR                 : mean reciprocal rank of the first relevant chunk
  - NDCG@k              : rank-discounted relevance
  - mean_similarity     : average dense cosine similarity of retrieved chunks to
                           the query (a pure embedding-quality signal, independent
                           of chunk-level ground truth)

If eval/eval_dataset.json does not exist, a small synthetic set is generated
automatically by sampling indexed chunks and asking the LLM to write a
question that chunk answers - a well-known bootstrap technique when no human
labels exist yet. Results are written to eval/results/<timestamp>.json and
eval/results/latest.json, plus a flattened CSV for spreadsheet review.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI

from src import config
from src.embeddings import get_embedding_function
from src.hybrid_retriever import build_hybrid_retriever
from src.vectorstore import get_all_documents


def _is_relevant(doc_meta: dict, expected: dict) -> bool:
    same_source = doc_meta.get("source") == expected.get("source")
    if "page" not in expected or expected["page"] is None:
        return same_source
    return same_source and int(doc_meta.get("page_num", -1)) == int(expected["page"])


def generate_synthetic_eval_set(n: int = 15, seed: int = 42) -> list[dict]:
    all_docs = get_all_documents()
    text_docs = [d for d in all_docs if d.metadata.get("content_type") == "text" and len(d.page_content) > 300]
    if not text_docs:
        text_docs = all_docs
    random.Random(seed).shuffle(text_docs)
    sample = text_docs[:n]

    llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.3)
    eval_set = []
    for doc in sample:
        prompt = (
            "Write ONE specific factual question that is answered by the passage below, "
            "the kind an equity analyst would ask about this bank's report. "
            "Reply with only the question, no preamble.\n\n"
            f"Passage:\n{doc.page_content[:1500]}"
        )
        try:
            question = llm.invoke(prompt).content.strip()
        except Exception as e:  # noqa: BLE001
            print(f"[evaluation] synthetic question generation failed: {e}")
            continue
        eval_set.append(
            {
                "question": question,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page_num"),
                "synthetic": True,
            }
        )

    config.EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
    print(f"[evaluation] wrote {len(eval_set)} synthetic Q/A pairs to {config.EVAL_DATASET_PATH}")
    return eval_set


def load_eval_set() -> list[dict]:
    if config.EVAL_DATASET_PATH.exists():
        with open(config.EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_synthetic_eval_set()


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    ideal = sorted(relevances, reverse=True)
    dcg, idcg = _dcg(relevances), _dcg(ideal)
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval(k: int = 5) -> dict:
    eval_set = load_eval_set()
    retriever = build_hybrid_retriever(top_n=k)
    embed_fn = get_embedding_function()

    rows = []
    for item in eval_set:
        question = item["question"]
        retrieved = retriever.invoke(question)[:k]
        relevances = [1 if _is_relevant(d.metadata, item) else 0 for d in retrieved]

        hit = int(any(relevances))
        precision = sum(relevances) / max(len(relevances), 1)
        total_relevant = 1  # single labeled chunk per question
        recall = min(sum(relevances), total_relevant) / total_relevant
        try:
            rank = relevances.index(1) + 1
            mrr = 1.0 / rank
        except ValueError:
            mrr = 0.0
        ndcg = _ndcg_at_k(relevances, k)

        query_vec = np.array(embed_fn.embed_query(question))
        sims = []
        if retrieved:
            doc_vecs = np.array(embed_fn.embed_documents([d.page_content for d in retrieved]))
            sims = (doc_vecs @ query_vec).tolist()  # vectors are L2-normalized -> dot == cosine
        mean_sim = float(np.mean(sims)) if sims else 0.0

        rows.append(
            {
                "question": question,
                "expected_source": item.get("source"),
                "expected_page": item.get("page"),
                "hit": hit,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": mrr,
                "ndcg_at_k": ndcg,
                "mean_similarity": mean_sim,
                "top_retrieved_source": retrieved[0].metadata.get("source") if retrieved else None,
                "top_retrieved_page": retrieved[0].metadata.get("page_num") if retrieved else None,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "k": k,
        "num_questions": len(df),
        "hit_rate": float(df["hit"].mean()) if len(df) else 0.0,
        "precision_at_k": float(df["precision_at_k"].mean()) if len(df) else 0.0,
        "recall_at_k": float(df["recall_at_k"].mean()) if len(df) else 0.0,
        "mrr": float(df["mrr"].mean()) if len(df) else 0.0,
        "ndcg_at_k": float(df["ndcg_at_k"].mean()) if len(df) else 0.0,
        "mean_similarity": float(df["mean_similarity"].mean()) if len(df) else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model": config.CHAT_MODEL,
        "reranker_enabled": config.USE_RERANKER,
    }

    _save_results(summary, df)
    return {"summary": summary, "rows": df.to_dict(orient="records")}


def _save_results(summary: dict, df: pd.DataFrame) -> None:
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"summary": summary, "rows": df.to_dict(orient="records")}

    with open(config.EVAL_RESULTS_DIR / f"{ts}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(config.EVAL_RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    df.to_csv(config.EVAL_RESULTS_DIR / f"{ts}.csv", index=False)
    df.to_csv(config.EVAL_RESULTS_DIR / "latest.csv", index=False)
    print(f"[evaluation] results saved to {config.EVAL_RESULTS_DIR / f'{ts}.json'}")


def load_latest_results() -> dict | None:
    path = config.EVAL_RESULTS_DIR / "latest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    result = evaluate_retrieval()
    print(json.dumps(result["summary"], indent=2))
