from sentence_transformers import SentenceTransformer
import numpy as np


class MMR:

    def __init__(self, model=None):
        self.model = model or SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    def select(self, query, results, top_k=10, lambda_param=0.7):

        documents = [
            result["document"]
            for result in results
        ]

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        document_embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        )

        relevance_scores = np.dot(
            document_embeddings,
            query_embedding
        )

        selected = []
        remaining = list(range(len(results)))

        while remaining and len(selected) < top_k:

            if not selected:
                best_index = max(
                    remaining,
                    key=lambda i: relevance_scores[i]
                )

            else:
                best_index = max(
                    remaining,
                    key=lambda i:
                    lambda_param * relevance_scores[i]
                    - (1 - lambda_param) *
                    max(
                        np.dot(
                            document_embeddings[i],
                            document_embeddings[j]
                        )
                        for j in selected
                    )
                )

            selected.append(best_index)
            remaining.remove(best_index)

        return [
            results[i]
            for i in selected
        ]