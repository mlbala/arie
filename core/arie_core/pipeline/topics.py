from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

TOPICS = {
    "ai": "Artificial intelligence",
    "llm": "LLMs",
    "agents": "Agents",
    "nlp": "Language / NLP",
    "vision": "Computer vision",
    "robotics": "Robotics",
    "ml": "Machine learning",
    "retrieval": "Retrieval / RAG",
    "safety": "Safety & alignment",
    "other": "Other research",
}
_PATTERNS = {
    "llm": r"\b(?:LLMs?|large language models?|language models?|transformers?|GPT[- ]?\d*)\b",
    "agents": r"\b(?:agents?|agentic|multi-agent|tool[- ]use)\b",
    "nlp": r"\b(?:natural language|NLP|machine translation|text generation)\b",
    "vision": r"\b(?:computer vision|image classification|object detection|visual recognition)\b",
    "robotics": r"\b(?:robots?|robotics|robotic|embodied)\b",
    "ml": r"\b(?:machine learning|deep learning|neural networks?|reinforcement learning)\b",
    "retrieval": r"\b(?:retrieval[- ]augmented|RAG|information retrieval|semantic search)\b",
    "safety": r"\b(?:alignment|AI safety|jailbreaks?|red[- ]teaming|model safety)\b",
}
_CATEGORIES = {
    "cs.AI": "ai",
    "cs.CL": "nlp",
    "cs.LG": "ml",
    "stat.ML": "ml",
    "cs.MA": "agents",
    "cs.CV": "vision",
    "cs.IR": "retrieval",
    "cs.NE": "ml",
}


def topic_tags(
    title: str,
    abstract: str,
    metadata: Mapping[str, str],
    pages: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Suggest topics from exact source matches, preserving their provenance; infer no claims."""
    tags: dict[str, dict[str, object]] = {}
    category = metadata.get("primary_category", "")
    if category in _CATEGORIES:
        key = _CATEGORIES[category]
        tags[key] = {
            "tag": key,
            "label": TOPICS[key],
            "basis": "source category",
            "quote": category,
        }
    sources: list[tuple[str, str, int | None]] = [
        ("title", title, None),
        ("abstract", abstract, None),
    ]
    sources.extend(("page", text[:20000], i + 1) for i, text in enumerate(pages[:5]))
    for key, pattern in _PATTERNS.items():
        for basis, text, page in sources:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tags[key] = {
                    "tag": key,
                    "label": TOPICS[key],
                    "basis": basis,
                    "quote": match.group(),
                    "page": page,
                }
                break
    if not tags:
        tags["other"] = {"tag": "other", "label": TOPICS["other"], "basis": "uncategorized"}
    return list(tags.values())
