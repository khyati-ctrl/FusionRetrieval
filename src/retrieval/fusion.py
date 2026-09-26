class RRFFusion:

    def __init__(self, k=60):
        self.k = k

    def fuse(self, result_lists, top_k=10):
        scores = {}
        documents = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):

                document_index = result["index"]

                rrf_score = 1 / (self.k + rank)

                scores[document_index] = (
                    scores.get(document_index, 0) + rrf_score
                )

                documents[document_index] = result["document"]

        ranked_indices = sorted(
            scores,
            key=scores.get,
            reverse=True
        )[:top_k]

        fused_results = []

        for index in ranked_indices:
            fused_results.append({
                "index": index,
                "score": scores[index],
                "document": documents[index]
            })

        return fused_results