"""
hyde.py
-------
Generates a *hypothetical code snippet* that would plausibly answer the
query, so we can search code-to-code instead of English-to-code.

Two modes:
  - "rule": template-based signature synthesis. Zero latency, zero
    dependencies, always works. This is your safety net and default.
  - "llm":  a small local model writes a short plausible snippet. Higher
    quality bridging, but adds latency and a failure mode — so it always
    falls back to "rule" mode on any error.

Design choice: we NEVER let a HyDE failure break the pipeline. If the LLM
isn't loaded, times out, or errors, we silently fall back to the rule-based
version rather than raising.
"""

from typing import List, Optional
from .analyzer import QueryAnalysis

# Maps intent -> a code "shape" template. {name} gets filled from
# identifiers/entities found in the query.
INTENT_TEMPLATES = {
    "how_to": "def {name}({args}):\n    \"\"\"{doc}\"\"\"\n    # implementation\n    pass",
    "what_is": "def {name}({args}):\n    \"\"\"{doc}\"\"\"\n    pass",
    "locate_definition": "def {name}({args}):\n    \"\"\"{doc}\"\"\"\n    pass",
    "usage_example": "result = {name}({args})",
    "debug_error": "try:\n    {name}({args})\nexcept Exception as e:\n    # {doc}\n    raise",
    "other": "def {name}({args}):\n    \"\"\"{doc}\"\"\"\n    pass",
}


class HydeGenerator:
    def __init__(self, llm_pipeline=None):
        """
        llm_pipeline: optional callable(prompt: str) -> str, e.g. a wrapped
        transformers/llama-cpp text-generation call. Left as None by default
        so this module has zero external dependencies until you wire one in.
        """
        self.llm_pipeline = llm_pipeline

    # --- rule-based path (always available) --------------------------
    def generate_rule_based(self, analysis: QueryAnalysis) -> str:
        template = INTENT_TEMPLATES.get(analysis.intent, INTENT_TEMPLATES["other"])

        name = analysis.identifiers[0] if analysis.identifiers else (
            analysis.entities[0] if analysis.entities else "target_function"
        )
        name = name.replace(" ", "_").lower()

        args = ", ".join(analysis.entities[:3]) if analysis.entities else "*args"
        doc = analysis.raw_query.strip().rstrip("?")

        return template.format(name=name, args=args, doc=doc)

    # --- LLM path (optional, falls back on any failure) ---------------
    def generate_llm(self, analysis: QueryAnalysis) -> Optional[str]:
        if self.llm_pipeline is None:
            return None
        prompt = (
            "Write a short, plausible code snippet (just the code, no "
            "explanation) that would answer this question about a codebase:\n"
            f"\"{analysis.raw_query}\"\n\nCode:\n"
        )
        try:
            output = self.llm_pipeline(prompt)
            return output.strip() if output else None
        except Exception:
            # Never let a generation failure break retrieval.
            return None

    def generate(self, analysis: QueryAnalysis, mode: str = "auto") -> str:
        """
        mode: "auto" (try llm, fall back to rule), "llm", or "rule".
        Always returns a usable string.
        """
        if mode in ("auto", "llm"):
            llm_result = self.generate_llm(analysis)
            if llm_result:
                return llm_result
            if mode == "llm":
                # explicit llm request but it failed -> still fall back,
                # don't return empty.
                pass
        return self.generate_rule_based(analysis)


if __name__ == "__main__":
    from .analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    hyde = HydeGenerator()  # no LLM wired in yet -> pure rule-based

    test_queries = [
        "how is the input validated before saving to the database",
        "where is the user authentication token generated",
        "why does the connection fail on retry",
    ]
    for q in test_queries:
        analysis = analyzer.analyze(q)
        snippet = hyde.generate(analysis)
        print(f"\nQuery: {q}")
        print(f"HyDE snippet:\n{snippet}")
