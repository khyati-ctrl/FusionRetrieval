"""
query_pipeline.py
------------------
The single entry point Member 3 (Retrieval) and Member 4 (API) will call.
Takes a raw NL query, runs it through analysis -> expansion -> HyDE, and
returns all three "views" of the query, ready to hand to retrieval.

This is deliberately the ONLY file other members need to import from.
Everything else in this package is an implementation detail behind it.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .analyzer import QueryAnalyzer, QueryAnalysis
from .expansion import QueryExpander
from .hyde import HydeGenerator


@dataclass
class MultiViewQuery:
    original: str
    expanded: str
    hyde: str
    intent: str
    entities: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "expanded": self.expanded,
            "hyde": self.hyde,
            "intent": self.intent,
            "entities": self.entities,
            "identifiers": self.identifiers,
        }

    def views_for_retrieval(self) -> List[str]:
        """
        The three strings that should each go through the retrieval /
        embedding step separately. Retrieval fuses their results (RRF).
        """
        return [self.original, self.expanded, self.hyde]


class QueryPipeline:
    def __init__(self, llm_pipeline=None):
        self.analyzer = QueryAnalyzer()
        self.expander = QueryExpander()
        self.hyde_gen = HydeGenerator(llm_pipeline=llm_pipeline)

    def process(self, query: str, hyde_mode: str = "auto") -> MultiViewQuery:
        analysis: QueryAnalysis = self.analyzer.analyze(query)
        expanded = self.expander.expand_query(analysis)
        hyde_snippet = self.hyde_gen.generate(analysis, mode=hyde_mode)

        return MultiViewQuery(
            original=analysis.raw_query,
            expanded=expanded,
            hyde=hyde_snippet,
            intent=analysis.intent,
            entities=analysis.entities,
            identifiers=analysis.identifiers,
        )


if __name__ == "__main__":
    pipeline = QueryPipeline()
    query = "how is the input validated before saving to the database"
    result = pipeline.process(query)

    print("Multi-view query representation:")
    for k, v in result.as_dict().items():
        print(f"  {k}: {v}")
