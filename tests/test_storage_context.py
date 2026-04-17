from app.services.storage import attach_neighbor_context


def test_attach_neighbor_context_expands_hit_with_adjacent_chunks():
    all_chunks = [
        {"chunk_id": "c1", "filename": "doc.pdf", "text": "Alice teaches this class."},
        {"chunk_id": "c2", "filename": "doc.pdf", "text": "She focuses on negotiation frameworks."},
        {"chunk_id": "c3", "filename": "doc.pdf", "text": "Final projects are team-based."},
    ]
    hits = [{"chunk_id": "c2", "filename": "doc.pdf", "text": "She focuses on negotiation frameworks."}]

    enriched = attach_neighbor_context(hits, all_chunks, window_size=1)

    assert len(enriched) == 1
    assert "Alice teaches this class." in enriched[0]["text"]
    assert "She focuses on negotiation frameworks." in enriched[0]["text"]
    assert "Final projects are team-based." in enriched[0]["text"]
