from __future__ import annotations

from collections import defaultdict

from app.utils.text import keyword_overlap_score, tokenize_for_bm25


def fuse_results(
    semantic_hits: list[dict],
    keyword_hits: list[dict],
    query: str,
    final_top_k: int = 5,
) -> list[dict]:
    combined: dict[str, dict] = {}

    semantic_max = max((hit["semantic_score"] for hit in semantic_hits), default=1.0) or 1.0
    keyword_max = max((hit["keyword_score"] for hit in keyword_hits), default=1.0) or 1.0
    query_terms = tokenize_for_bm25(query)

    for hit in semantic_hits:
        item = dict(hit)
        item["semantic_rank"] = hit.get("rank", 0)
        item["semantic_norm"] = hit["semantic_score"] / semantic_max
        item.setdefault("keyword_score", 0.0)
        combined[item["chunk_id"]] = item

    for hit in keyword_hits:
        existing = combined.get(hit["chunk_id"], dict(hit))
        existing["keyword_rank"] = hit.get("rank", 0)
        existing["keyword_score"] = hit["keyword_score"]
        existing["keyword_norm"] = hit["keyword_score"] / keyword_max
        combined[hit["chunk_id"]] = existing

    by_text = defaultdict(list)
    for item in combined.values():
        by_text[item["text"].strip().lower()].append(item["chunk_id"])

    for item in combined.values():
        semantic_norm = item.get("semantic_norm", 0.0)
        keyword_norm = item.get("keyword_norm", 0.0)
        overlap = keyword_overlap_score(query_terms, item["text"])
        duplicate_penalty = 0.08 if len(by_text[item["text"].strip().lower()]) > 1 else 0.0

        item["final_score"] = (
            semantic_norm * 0.55
            + keyword_norm * 0.35
            + overlap * 0.18
            - duplicate_penalty
        )

    ranked = sorted(combined.values(), key=lambda item: item["final_score"], reverse=True)
    return ranked[:final_top_k]


def evidence_is_strong(hits: list[dict]) -> bool:
    if not hits:
        return False

    scores = [hit.get("final_score", 0.0) or 0.0 for hit in hits]
    top = scores[0]
    strong_count = sum(1 for s in scores if s >= 0.35)

    if top >= 0.5:
        return True
    if strong_count >= 2:
        return True
    return False
