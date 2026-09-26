queries = [
    "find code that calculates the mean of numbers",
    "find the maximum number",
    "reverse a string",
    "sort numbers",
    "count words in text"
]


query_pipeline = QueryPipeline()
retrieval_pipeline = RetrievalPipeline(documents)


for query in queries:

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    query_result = query_pipeline.process(query)

    query_views = query_result.views_for_retrieval()

    results = retrieval_pipeline.search(
        query_views,
        rrf_top_k=5,
        rerank_top_k=3,
        final_top_k=2
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Index: {result['index']}")
        print(f"Document: {result['document']}")