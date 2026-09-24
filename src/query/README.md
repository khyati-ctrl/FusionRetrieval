# Member 2 — Query Intelligence / Vocabulary Bridge

## What's here

- `analyzer.py` — intent detection, entity extraction, identifier extraction (rule-based, instant)
- `expansion.py` — synonym + identifier-variant query expansion
- `hyde.py` — hypothetical code snippet generation (rule-based by default; pluggable LLM)
- `query_pipeline.py` — **the single entry point**. Combines the above into a `MultiViewQuery`.
- `benchmark.py` — compares original vs expanded vs HyDE vs combined (RRF-fused) retrieval quality

## Quick start

```bash
python3 -m src.query.query_pipeline     # see the full pipeline output
python3 -m src.query.benchmark          # run the toy benchmark (validates logic only)
```

## For Member 3 (Retrieval) / Member 4 (API)

Import only this:

```python
from src.query.query_pipeline import QueryPipeline

pipeline = QueryPipeline()
mv = pipeline.process("how is the input validated before saving")

mv.original      # raw query string
mv.expanded       # query + synonyms + identifier variants
mv.hyde            # hypothetical code snippet
mv.views_for_retrieval()  # [original, expanded, hyde] — search each, then RRF-fuse
```

## Status / what's real vs. stubbed

| Piece | Status |
|---|---|
| Intent detection | Rule-based, working, tested on sample queries |
| Entity extraction | Rule-based with a small technical vocabulary — **extend `TECHNICAL_ENTITIES` in `analyzer.py` as you see real query patterns from the actual dataset** |
| Identifier extraction/generation | Rule-based, working |
| Synonym expansion | Small hand-built dict in `expansion.py` — **extend `SYNONYMS` based on real codebase vocabulary** |
| HyDE (rule-based) | Working, template-based, zero dependencies |
| HyDE (LLM-based) | **Not wired in yet.** `HydeGenerator(llm_pipeline=...)` accepts any `callable(prompt) -> str`. Plug in a small local model (via `transformers` or `llama-cpp-python`) here once you've picked one and benchmarked the latency. |
| Benchmark | Logic validated against a toy 5-chunk example. **Needs Member 1's real `embed_fn`/`search_fn` and a real labeled eval set to produce meaningful numbers.** |

## Your next concrete steps, in order

1. **Get a labeled mini eval set.** Even 15-20 (query, relevant_chunk_ids) pairs from your actual target repo, hand-labeled, is enough to start seeing real signal in the benchmark.
2. **Plug in the real embedding model.** Once Member 1 has EmbeddingGemma wired up, swap `fake_embed_fn`/`fake_search_fn` in `benchmark.py` for calls into their pipeline.
3. **Run the benchmark on real data** and look at the table: does `combined` actually beat `original` alone? Does `hyde` alone ever hurt? This is the evidence you bring to the team about which strategy to ship.
4. **Only after that**, evaluate whether it's worth the added latency to wire in an LLM-based HyDE instead of the rule-based template version — benchmark both if time allows.
5. **Extend the vocabulary** (`TECHNICAL_ENTITIES`, `SYNONYMS`) based on what your actual target codebase looks like — the current lists are generic starting points, not tuned to your specific repo.

## Coordination needed with Member 1

You need to agree on the **chunk ID format** early — `benchmark.py`'s `search_fn` assumes it gets back a list of chunk IDs matching whatever Member 1 uses as the primary key in LanceDB.
