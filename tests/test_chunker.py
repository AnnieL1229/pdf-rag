from app.services.chunker import TextChunker


def test_chunker_respects_max_length():
    chunker = TextChunker(max_chars=80, overlap=10)
    text = (
        "Paragraph one is short.\n\n"
        "Paragraph two is a bit longer and includes several sentences. "
        "It should cause the chunker to split while still keeping the output readable.\n\n"
        "Paragraph three wraps things up."
    )

    chunks = chunker.chunk_page(text)

    assert chunks
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_chunker_uses_overlap_for_long_text():
    chunker = TextChunker(max_chars=60, overlap=12)
    text = "A" * 140

    chunks = chunker.chunk_page(text)

    assert len(chunks) >= 2
    assert chunks[0][-12:] == chunks[1][:12]
