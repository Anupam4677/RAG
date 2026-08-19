"""Gradio UI for the bank-reports RAG system.

Three tabs:
  - Chat: ask questions, see grounded answers with source/page citations
  - Ingest: trigger ingestion of PDFs from data_pdf_files/ and watch progress
  - Evaluation: run the retrieval/embedding evaluation and view results
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import gradio as gr
import pandas as pd

from src import config
from src.ingest import discover_pdfs, ingest_file
from src.rag_chain import get_rag_chain
from src.vectorstore import collection_count


def chat_fn(message: str, history: list[dict]):
    chain = get_rag_chain()
    pairs = []
    for i in range(0, len(history) - 1, 2):
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i]["content"], history[i + 1]["content"]))

    result = chain.answer(message, pairs)

    sources_md = "\n\n**Sources retrieved:**\n"
    for s in result.sources:
        sources_md += (
            f"- `{s['source']}` p.{s['page']} "
            f"({s['content_type']}, section: {s['section'] or 'N/A'}) — "
            f"_{s['snippet'].strip()}..._\n"
        )
    return result.answer + sources_md


def ingest_fn(progress=gr.Progress()):
    pdfs = discover_pdfs()
    if not pdfs:
        return f"No PDFs found in {config.DATA_DIR}"

    log_buffer = io.StringIO()
    total = 0
    for i, path in enumerate(pdfs):
        progress((i, len(pdfs)), desc=f"Ingesting {path.name}")
        with redirect_stdout(log_buffer):
            try:
                total += ingest_file(path)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED on {path.name}: {e}")

    get_rag_chain().refresh_retriever()
    log_buffer.write(f"\nDone. {total} chunks upserted. Collection now holds {collection_count()} chunks.\n")
    return log_buffer.getvalue()


def eval_fn(k: int, progress=gr.Progress()):
    from src.evaluation import evaluate_retrieval

    progress((0, 1), desc="Running evaluation...")
    result = evaluate_retrieval(k=int(k))
    progress((1, 1), desc="Done")
    summary_df = pd.DataFrame([result["summary"]])
    rows_df = pd.DataFrame(result["rows"])
    return summary_df, rows_df, json.dumps(result["summary"], indent=2)


def load_latest_eval_fn():
    from src.evaluation import load_latest_results

    result = load_latest_results()
    if result is None:
        return pd.DataFrame(), pd.DataFrame(), "No evaluation results yet. Run an evaluation first."
    summary_df = pd.DataFrame([result["summary"]])
    rows_df = pd.DataFrame(result["rows"])
    return summary_df, rows_df, json.dumps(result["summary"], indent=2)


def latency_profile_fn(n_questions: int, progress=gr.Progress()):
    from src.latency_profiler import load_profile_questions, run_latency_profile

    questions = load_profile_questions()[: int(n_questions)]
    progress((0, 1), desc=f"Profiling {len(questions)} questions...")
    df = run_latency_profile(questions)
    progress((1, 1), desc="Done")

    stage_cols = [c for c in df.columns if c.endswith("_ms")]
    summary_df = df[stage_cols].mean().to_frame(name="mean_ms").reset_index(names="stage") if len(df) else pd.DataFrame()
    return summary_df, df


def load_latest_latency_fn():
    from src.latency_profiler import load_latest_results

    df = load_latest_results()
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    stage_cols = [c for c in df.columns if c.endswith("_ms")]
    summary_df = df[stage_cols].mean().to_frame(name="mean_ms").reset_index(names="stage")
    return summary_df, df


with gr.Blocks(title="Bank Reports RAG") as demo:
    gr.Markdown("# 🏦 Bank Annual Reports — RAG Assistant")
    gr.Markdown(
        "Ask questions about the bank annual reports, investor presentations, "
        "and quarterly results indexed from `data_pdf_files/`. Answers are grounded "
        "in retrieved chunks with source + page citations."
    )

    with gr.Tab("Chat"):
        gr.ChatInterface(fn=chat_fn)

    with gr.Tab("Ingest"):
        gr.Markdown(
            "Parses every PDF in `data_pdf_files/`, chunks it, enriches metadata, "
            "embeds it, and upserts it into the Chroma vector store. Re-running is "
            "safe — chunk IDs are deterministic, so re-ingesting the same file "
            "overwrites its existing chunks instead of duplicating them."
        )
        ingest_btn = gr.Button("Run Ingestion", variant="primary")
        ingest_log = gr.Textbox(label="Ingestion log", lines=20)
        ingest_btn.click(fn=ingest_fn, outputs=ingest_log)

    with gr.Tab("Evaluation"):
        gr.Markdown(
            "Evaluates retrieval quality (Hit Rate, Precision/Recall@k, MRR, NDCG) "
            "and embedding quality (mean query-chunk cosine similarity) against "
            "`eval/eval_dataset.json`. If that file doesn't exist yet, a synthetic "
            "eval set is generated automatically from the indexed corpus."
        )
        with gr.Row():
            k_slider = gr.Slider(1, 10, value=5, step=1, label="k (top-k retrieved)")
            run_eval_btn = gr.Button("Run Evaluation", variant="primary")
            load_eval_btn = gr.Button("Load Latest Results")
        eval_summary = gr.Dataframe(label="Summary")
        eval_rows = gr.Dataframe(label="Per-question results")
        eval_json = gr.Code(label="Summary JSON", language="json")

        run_eval_btn.click(fn=eval_fn, inputs=k_slider, outputs=[eval_summary, eval_rows, eval_json])
        load_eval_btn.click(fn=load_latest_eval_fn, outputs=[eval_summary, eval_rows, eval_json])

    with gr.Tab("Latency"):
        gr.Markdown(
            "Breaks down end-to-end query latency into per-component stages "
            "(query processing, embedding generation, vector search, BM25 search, "
            "reranking, LLM time-to-first-token, LLM total generation) across a "
            "sample of eval questions. Results are saved to `eval/latency_results/`."
        )
        with gr.Row():
            n_slider = gr.Slider(1, 20, value=5, step=1, label="Number of questions to profile")
            run_latency_btn = gr.Button("Run Latency Profile", variant="primary")
            load_latency_btn = gr.Button("Load Latest Results")
        latency_summary = gr.Dataframe(label="Mean latency per stage (ms)")
        latency_rows = gr.Dataframe(label="Per-question latency breakdown")

        run_latency_btn.click(fn=latency_profile_fn, inputs=n_slider, outputs=[latency_summary, latency_rows])
        load_latency_btn.click(fn=load_latest_latency_fn, outputs=[latency_summary, latency_rows])


if __name__ == "__main__":
    demo.launch()
