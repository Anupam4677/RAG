# Changes

## 2026-08-15 — Streaming chat responses

**What changed**

- `src/rag_chain.py`
  - Extracted the shared "retrieve docs -> build messages + sources" logic out of
    `RAGChain.answer()` into a new private helper, `_build_messages_and_sources()`,
    so both the streaming and non-streaming paths retrieve context exactly once
    and stay in sync.
  - `RAGChain.answer()` is unchanged in behavior — still does a single
    non-streaming `llm.invoke(...)` call and returns a complete `RAGAnswer`.
    (Used by the CLI one-liner in the README/CLAUDE.md and anywhere a full
    answer is needed synchronously.)
  - Added `RAGChain.stream_answer(question, history)` — a generator that calls
    `llm.stream(...)` and yields `(partial_answer_text, sources)` on every
    token/chunk received from the model. `sources` is constant across yields
    (retrieval happens once, up front, before generation starts); only
    `partial_answer_text` grows as tokens arrive.

- `app.py`
  - `chat_fn` is now a generator function instead of a plain function. It
    iterates `chain.stream_answer(...)`, `yield`-ing the growing answer text
    so Gradio's `ChatInterface` renders tokens incrementally in the Chat tab
    instead of waiting for the full response.
  - The "Sources retrieved" citation block is appended only after the token
    stream finishes, in one final `yield`, so citations don't flicker in in
    the middle of streaming.

**Why**

The Chat tab previously blocked until the full LLM response was generated,
which feels slow for longer, multi-figure financial answers. Streaming gives
immediate visual feedback (tokens appear as the model produces them) with no
change to retrieval, grounding, or citation behavior.

**Not changed**

- Retrieval (hybrid dense + BM25 + rerank) — identical for both `answer()`
  and `stream_answer()`, since both go through `_build_messages_and_sources()`.
- The system prompt / citation-mandatory grounding rules.
- The CLI usage pattern (`get_rag_chain().answer(...)`) — still works exactly
  as before, still non-streaming.
- Ingest and Evaluation tabs — untouched.

**How to verify**

```powershell
python app.py
```

Open the Chat tab and ask a question — the answer should now appear
incrementally (token-by-token / chunk-by-chunk) rather than all at once, with
the sources list appearing once streaming completes.
