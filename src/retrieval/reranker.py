from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=20):

        pairs = []

        for result in results:
            pairs.append(
                [query, result["document"]]
            )

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append({
                "index": result["index"],
                "score": float(score),
                "document": result["document"]
            })

        reranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked_results[:top_k]