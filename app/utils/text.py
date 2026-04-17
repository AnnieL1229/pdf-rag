import re
from collections.abc import Iterable


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    if not text.strip():
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_overlap_score(query_terms: Iterable[str], text: str) -> float:
    lowered = text.lower()
    query_terms = [term for term in query_terms if term]
    if not query_terms:
        return 0.0
    hits = sum(1 for term in query_terms if term in lowered)
    return hits / len(query_terms)
