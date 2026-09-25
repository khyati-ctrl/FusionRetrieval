from sentence_transformers import SentenceTransformer
import numpy as np


class DenseRetriever:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

    def search(self, query, documents, top_k=5):
        query_embedding = self.encode([query])[0]
        document_embeddings = self.encode(documents)

        scores = np.dot(document_embeddings, query_embedding)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "index": int(index),
                "score": float(scores[index]),
                "document": documents[index]
            })

        return results