"""
run_real_benchmark.py
-----------------------
Connects everything you've built into one real run:
  1. Real labeled data from MTEB (load_mteb_eval_set_fixed.py)
  2. Your query pipeline (Original / Expanded / HyDE) (query_pipeline.py)
  3. A real embedding model (sentence-transformers) for actual similarity search
  4. Your benchmark comparison logic (benchmark.py)

Run this from your repo root:
    python scripts/run_real_benchmark.py

Requires (should already be installed from earlier):
    pip install sentence-transformers

NOTE ON SPEED: embedding all 8765 corpus docs on CPU on a first run can be
slow. This script samples a smaller subset by default (CORPUS_SAMPLE_SIZE)
guaranteed to include every correct answer from your eval set, plus random
extra documents as "distractors" -- so the benchmark is still meaningful,
just faster to iterate on. Raise CORPUS_SAMPLE_SIZE once this is confirmed
working end-to-end.
"""

import sys
import random
from pathlib import Path

# Make sure we can import from src/query/ regardless of where this is run from
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.query.query_pipeline import QueryPipeline
from src.query.benchmark import RetrievalBenchmark
from src.query.load_mteb_eval_set_fixed import load_mteb_task, to_benchmark_eval_set, get_corpus_lookup

MAX_QUERIES = 10          # how many labeled queries to evaluate on
CORPUS_SAMPLE_SIZE = 500  # how many corpus docs to embed (for speed on first run)
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, well-tested -- swap for EmbeddingGemma later


def build_corpus_sample(corpus: dict, eval_set, sample_size: int) -> dict:
    """Guarantee every correct answer is included, then fill up to sample_size randomly."""
    required_ids = set()
    for _, relevant_ids in eval_set:
        required_ids.update(relevant_ids)

    required_ids = {rid for rid in required_ids if rid in corpus}
    remaining_ids = [cid for cid in corpus.keys() if cid not in required_ids]

    extra_needed = max(0, sample_size - len(required_ids))
    random.seed(42)  # reproducible sampling
    sampled_extra = random.sample(remaining_ids, min(extra_needed, len(remaining_ids)))

    final_ids = list(required_ids) + sampled_extra
    return {cid: corpus[cid] for cid in final_ids}


def main():
    print(f"Loading embedding model: {MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print("Loading MTEB task data...")
    task = load_mteb_task()
    eval_set = to_benchmark_eval_set(task, max_queries=MAX_QUERIES)
    full_corpus = get_corpus_lookup(task)

    print(f"Sampling corpus down to ~{CORPUS_SAMPLE_SIZE} docs (guaranteeing correct answers are included)...")
    corpus_sample = build_corpus_sample(full_corpus, eval_set, CORPUS_SAMPLE_SIZE)
    corpus_ids = list(corpus_sample.keys())
    corpus_texts = [corpus_sample[cid] for cid in corpus_ids]

    print(f"Embedding {len(corpus_texts)} corpus documents (this may take a minute)...")
    corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)

    import numpy as np
    from numpy.linalg import norm

    corpus_norms = norm(corpus_embeddings, axis=1, keepdims=True)
    corpus_normalized = corpus_embeddings / np.clip(corpus_norms, 1e-10, None)

    def embed_fn(text: str):
        vec = model.encode([text], convert_to_numpy=True)[0]
        return vec / max(norm(vec), 1e-10)

    def search_fn(query_vector, k: int):
        sims = corpus_normalized @ query_vector  # cosine similarity since both normalized
        top_k_idx = np.argsort(-sims)[:k]
        return [corpus_ids[i] for i in top_k_idx]

    print("Running benchmark: Original vs Expanded vs HyDE vs Combined (equal weights)...\n")
    pipeline = QueryPipeline()
    bench_equal = RetrievalBenchmark(embed_fn=embed_fn, search_fn=search_fn, pipeline=pipeline)
    results_equal = bench_equal.run(eval_set, k=10)
    bench_equal.print_report(results_equal)

    print("\nRunning again with HyDE weighted 2x in the fusion...\n")
    bench_weighted = RetrievalBenchmark(
        embed_fn=embed_fn, search_fn=search_fn, pipeline=pipeline,
        fusion_weights=[1.0, 1.0, 2.0],  # [original, expanded, hyde]
    )
    results_weighted = bench_weighted.run(eval_set, k=10)
    bench_weighted.print_report(results_weighted)

    print("\n--- Summary: does weighting HyDE higher fix 'combined'? ---")
    print(f"{'Strategy':<12} {'Equal NDCG':>12} {'Weighted NDCG':>15}")
    for strategy in ["original", "expanded", "hyde", "combined"]:
        print(f"{strategy:<12} {results_equal[strategy]['ndcg@10']:>12.3f} "
              f"{results_weighted[strategy]['ndcg@10']:>15.3f}")


if __name__ == "__main__":
    main()
