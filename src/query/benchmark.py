"""
benchmark.py
------------
Your critical responsibility: prove which query representation(s) actually
help retrieval, rather than assuming.

This compares 4 strategies:
  1. original          - raw NL query only
  2. expanded           - original + synonym/identifier expansion
  3. hyde                - HyDE hypothetical-snippet only
  4. combined (RRF)      - original + expanded + hyde, fused

HOW TO USE THIS FOR REAL:
This file is deliberately decoupled from any specific embedding model or
index. You plug in:
  - `embed_fn(text) -> vector`      (from Member 1's embedding model)
  - `search_fn(query_vector, k) -> List[chunk_id]`  (from Member 1's index)
and a small labeled eval set of (query, [relevant_chunk_ids]) pairs.

Until the real index exists, `run_demo()` uses a tiny synthetic toy example
so you can verify the benchmarking LOGIC is correct before plugging in
real data. Swap in real embed_fn/search_fn/eval_set and nothing else here
needs to change.
"""

import math
from typing import Callable, Dict, List, Tuple
from collections import defaultdict

from .query_pipeline import QueryPipeline


# --- metrics --------------------------------------------------------------

def dcg_at_k(relevances: List[int], k: int) -> float:
    return sum(
        rel / math.log2(idx + 2)  # idx+2 because idx starts at 0 -> log2(2)
        for idx, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    relevances = [1 if cid in relevant_ids else 0 for cid in ranked_ids]
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(ranked_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


# --- RRF fusion (used by the "combined" strategy) --------------------------

def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60,
                            weights: List[float] = None) -> List[str]:
    """
    Weighted RRF: score = sum over lists of weight_i / (k + rank).
    weights defaults to equal weighting (the original behavior) if not given.
    Pass e.g. weights=[1.0, 1.0, 2.0] to trust the 3rd list (HyDE) twice as
    much as the others -- use this when a benchmark run shows one view is
    consistently stronger, so consensus among weaker views can't drown it out.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] += weight / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


# --- benchmark runner -------------------------------------------------------

EmbedFn = Callable[[str], "Any"]
SearchFn = Callable[["Any", int], List[str]]


class RetrievalBenchmark:
    def __init__(self, embed_fn: EmbedFn, search_fn: SearchFn, pipeline: QueryPipeline = None,
                 fusion_weights: List[float] = None):
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.pipeline = pipeline or QueryPipeline()
        # [original_weight, expanded_weight, hyde_weight]. None = equal weighting.
        self.fusion_weights = fusion_weights

    def _search_view(self, text: str, k: int = 50) -> List[str]:
        vec = self.embed_fn(text)
        return self.search_fn(vec, k)

    def evaluate_query(self, query: str, relevant_ids: set, k: int = 10) -> Dict[str, Dict[str, float]]:
        mv = self.pipeline.process(query)

        results_original = self._search_view(mv.original)
        results_expanded = self._search_view(mv.expanded)
        results_hyde = self._search_view(mv.hyde)
        results_combined = reciprocal_rank_fusion(
            [results_original, results_expanded, results_hyde],
            weights=self.fusion_weights,
        )

        strategies = {
            "original": results_original,
            "expanded": results_expanded,
            "hyde": results_hyde,
            "combined": results_combined,
        }

        return {
            name: {
                "ndcg@10": ndcg_at_k(ranked, relevant_ids, k),
                "recall@10": recall_at_k(ranked, relevant_ids, k),
            }
            for name, ranked in strategies.items()
        }

    def run(self, eval_set: List[Tuple[str, set]], k: int = 10) -> Dict[str, Dict[str, float]]:
        """eval_set: list of (query, set_of_relevant_chunk_ids)."""
        totals = defaultdict(lambda: defaultdict(list))

        for query, relevant_ids in eval_set:
            per_query = self.evaluate_query(query, relevant_ids, k)
            for strategy, metrics in per_query.items():
                for metric_name, value in metrics.items():
                    totals[strategy][metric_name].append(value)

        averaged = {
            strategy: {m: sum(vals) / len(vals) for m, vals in metrics.items()}
            for strategy, metrics in totals.items()
        }
        return averaged

    def print_report(self, results: Dict[str, Dict[str, float]]):
        print(f"{'Strategy':<12} {'NDCG@10':>10} {'Recall@10':>12}")
        print("-" * 36)
        for strategy in ["original", "expanded", "hyde", "combined"]:
            m = results[strategy]
            print(f"{strategy:<12} {m['ndcg@10']:>10.3f} {m['recall@10']:>12.3f}")


# --- toy demo so you can validate the logic before real data exists -------

def run_demo():
    """
    Tiny synthetic example: a fake 'index' of 5 chunks, where we know which
    ones are relevant to each toy query. This proves the benchmark logic
    works correctly. Replace embed_fn/search_fn/eval_set with real ones
    once Member 1's index is ready -- nothing else changes.
    """
    fake_chunks = {
        "c1": "def validate_input(data): check schema and sanitize",
        "c2": "def save_to_db(record): persist record to database",
        "c3": "def get_user(id): fetch user account by id",
        "c4": "def generate_token(user): create auth token for login",
        "c5": "def retry_connection(): reconnect on socket failure",
    }

    def fake_embed_fn(text: str):
        # toy "embedding": just keep the lowercase text, real version
        # would call EmbeddingGemma / sentence-transformers here.
        return text.lower()

    def fake_search_fn(query_text: str, k: int):
        # toy "search": naive word-overlap scoring, real version would
        # do vector similarity search against LanceDB.
        q_words = set(query_text.split())
        scored = []
        for cid, text in fake_chunks.items():
            overlap = len(q_words & set(text.lower().split()))
            scored.append((cid, overlap))
        scored.sort(key=lambda x: -x[1])
        return [cid for cid, _ in scored[:k]]

    eval_set = [
        ("how is the input validated before saving to the database", {"c1", "c2"}),
        ("where is the user authentication token generated", {"c3", "c4"}),
        ("why does the connection fail on retry", {"c5"}),
    ]

    bench = RetrievalBenchmark(embed_fn=fake_embed_fn, search_fn=fake_search_fn)
    results = bench.run(eval_set)
    bench.print_report(results)
    return results


if __name__ == "__main__":
    run_demo()
