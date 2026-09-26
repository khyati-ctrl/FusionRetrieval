import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            self.tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def tokenize(self, text):
        text = text.lower()

        tokens = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            text
        )

        expanded_tokens = []

        for token in tokens:
            expanded_tokens.append(token)

            if "_" in token:
                expanded_tokens.extend(token.split("_"))

        return expanded_tokens

    def search(self, query, top_k=5):
        tokenized_query = self.tokenize(query)

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