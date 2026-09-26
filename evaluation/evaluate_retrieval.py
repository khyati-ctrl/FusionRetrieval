import json

from src.query.query_pipeline import QueryPipeline
from src.retrieval.pipeline import RetrievalPipeline


# Load chunks
chunks = []

with open("data/chunks/chunks.jsonl", "r") as file:
    for line in file:
        chunks.append(json.loads(line))


# Retrieval documents
documents = [
    chunk["augmented_text"]
    for chunk in chunks
]


# Expected answer for each query
evaluation_queries = [
    {
        "query": "find code that calculates the total",
        "expected_function": "calculate_total"
    },
    {
        "query": "find code that verifies a token",
        "expected_function": "verify_token"
    },
    {
        "query": "find code that logs out a user",
        "expected_function": "logout"
    },
    {
        "query": "find code that normalizes text",
        "expected_function": "normalize_input"
    }
]


query_pipeline = QueryPipeline()
retrieval_pipeline = RetrievalPipeline(documents)

# Run evaluation
total = len(evaluation_queries)

recall_at_1 = 0
recall_at_2 = 0
reciprocal_rank_sum = 0


# Map function names to their chunk indexes
expected_indices = {
    chunk["function_name"]: index
    for index, chunk in enumerate(chunks)
}


for item in evaluation_queries:

    query = item["query"]
    expected_function = item["expected_function"]

    expected_index = expected_indices[expected_function]

    query_result = query_pipeline.process(query)

    results = retrieval_pipeline.search(
        query_result.views_for_retrieval(),
        rrf_top_k=50,
        rerank_top_k=20,
        final_top_k=10
    )

    retrieved_indices = [
        result["index"]
        for result in results
    ]

    # Recall@1
    if expected_index in retrieved_indices[:1]:
        recall_at_1 += 1

    # Recall@2
    if expected_index in retrieved_indices[:2]:
        recall_at_2 += 1

    # Reciprocal Rank
    if expected_index in retrieved_indices:
        rank = retrieved_indices.index(expected_index) + 1
        reciprocal_rank_sum += 1 / rank

    print("\nQuery:", query)
    print("Expected:", expected_function)
    print("Retrieved:", retrieved_indices)


# Final metrics
recall_at_1_score = recall_at_1 / total
recall_at_2_score = recall_at_2 / total
mrr_score = reciprocal_rank_sum / total


print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

print(f"Recall@1: {recall_at_1_score:.2f}")
print(f"Recall@2: {recall_at_2_score:.2f}")
print(f"MRR:      {mrr_score:.2f}")