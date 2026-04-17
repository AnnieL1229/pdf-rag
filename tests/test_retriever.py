from app.services.retriever import evidence_is_strong, fuse_results


def test_fuse_results_prefers_combined_hits():
    semantic_hits = [
        {
            "chunk_id": "a",
            "filename": "doc.pdf",
            "page_number": 1,
            "text": "The refund policy allows returns within 30 days.",
            "semantic_score": 0.9,
            "rank": 1,
        },
        {
            "chunk_id": "b",
            "filename": "doc.pdf",
            "page_number": 2,
            "text": "Office locations are listed here.",
            "semantic_score": 0.8,
            "rank": 2,
        },
    ]
    keyword_hits = [
        {
            "chunk_id": "a",
            "filename": "doc.pdf",
            "page_number": 1,
            "text": "The refund policy allows returns within 30 days.",
            "keyword_score": 3.0,
            "rank": 1,
        }
    ]

    results = fuse_results(semantic_hits, keyword_hits, query="refund policy", final_top_k=2)

    assert results[0]["chunk_id"] == "a"
    assert results[0]["final_score"] > results[1]["final_score"]


def test_evidence_is_strong_when_top_score_is_high():
    hits = [{"chunk_id": "a", "final_score": 0.52}, {"chunk_id": "b", "final_score": 0.1}]

    assert evidence_is_strong(hits) is True


def test_evidence_is_strong_when_two_moderate_chunks_exist():
    hits = [{"chunk_id": "a", "final_score": 0.36}, {"chunk_id": "b", "final_score": 0.35}]

    assert evidence_is_strong(hits) is True


def test_evidence_is_not_strong_when_scores_are_weak():
    hits = [{"chunk_id": "a", "final_score": 0.34}, {"chunk_id": "b", "final_score": 0.2}]

    assert evidence_is_strong(hits) is False
