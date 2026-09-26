from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            document.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=5):
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "index": int(index),
                "score": float(scores[index]),
                "document": self.documents[index]
            })

        return results