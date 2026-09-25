"""
expansion.py
------------
Takes the raw query + the analysis from analyzer.py, and produces an
*expanded* version of the query: more words that are likely to actually
appear in code (synonyms real developers use in identifiers/comments),
plus identifier-style variants.

Why this matters: a query says "validate", the code says "check" or
"verify" or "is_valid". Plain embedding similarity alone doesn't reliably
catch this — an explicit synonym table does, cheaply and deterministically.
"""

from typing import List
from .analyzer import QueryAnalysis

# Hand-built technical synonym map. Extend this as you observe real gaps
# between query wording and codebase wording during testing.
SYNONYMS = {
    "validate": ["check", "verify", "is_valid", "sanitize"],
    "fetch": ["get", "retrieve", "load", "read"],
    "save": ["store", "persist", "write", "write_to"],
    "delete": ["remove", "destroy", "drop", "clear"],
    "create": ["make", "build", "init", "new"],
    "update": ["modify", "edit", "set", "patch"],
    "error": ["exception", "failure", "fault"],
    "input": ["arg", "argument", "param", "parameter", "payload"],
    "output": ["result", "response", "return_value"],
    "user": ["account", "member", "profile"],
    "authentication": ["auth", "login", "signin"],
    "authorization": ["auth", "permission", "access_control"],
    "connect": ["connection", "link", "establish"],
    "send": ["emit", "dispatch", "publish", "post"],
    "receive": ["consume", "subscribe", "listen"],
    "generate": ["create", "produce", "build"],
    "encrypt": ["encode", "cipher", "hash"],
    "decrypt": ["decode", "unhash"],
}


class QueryExpander:
    def expand_terms(self, analysis: QueryAnalysis) -> List[str]:
        """Return extra terms (not the original query) worth adding."""
        extra = []
        query_lower = analysis.raw_query.lower()

        for entity in analysis.entities:
            if entity in SYNONYMS:
                for syn in SYNONYMS[entity]:
                    if syn not in query_lower:
                        extra.append(syn)

        # de-dupe, preserve order
        seen, out = set(), []
        for term in extra:
            if term not in seen:
                seen.add(term)
                out.append(term)
        return out

    def expand_query(self, analysis: QueryAnalysis) -> str:
        """
        Build one expanded query string: original text + synonym terms +
        identifier variants. This is what gets embedded/searched as the
        'expanded' view.
        """
        extra_terms = self.expand_terms(analysis)
        parts = [analysis.raw_query] + extra_terms + analysis.identifiers
        return " ".join(parts)


if __name__ == "__main__":
    from .analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    expander = QueryExpander()

    test_queries = [
        "how is the input validated before saving to the database",
        "where is the user authentication token generated",
    ]
    for q in test_queries:
        analysis = analyzer.analyze(q)
        expanded = expander.expand_query(analysis)
        print(f"\nOriginal: {q}")
        print(f"Expanded: {expanded}")
