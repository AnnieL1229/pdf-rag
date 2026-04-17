from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import faiss
from rank_bm25 import BM25Okapi

from app.core.config import DATA_DIR, settings
from app.services.chunker import TextChunker
from app.services.embedder import Embedder
from app.services.pdf_parser import PDFParser
from app.services.retriever import fuse_results
from app.utils.text import tokenize_for_bm25


def attach_neighbor_context(
    hits: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    window_size: int = 1,
) -> list[dict[str, Any]]:
    if window_size <= 0 or not hits or not all_chunks:
        return [dict(hit) for hit in hits]

    chunks_by_filename: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, chunk in enumerate(all_chunks):
        filename = str(chunk.get("filename", ""))
        if filename:
            chunks_by_filename.setdefault(filename, []).append((idx, chunk))

    for filename in chunks_by_filename:
        chunks_by_filename[filename].sort(key=lambda pair: pair[0])

    contextual_hits: list[dict[str, Any]] = []
    for hit in hits:
        enriched = dict(hit)
        chunk_id = str(hit.get("chunk_id", ""))
        filename = str(hit.get("filename", ""))
        if not chunk_id or not filename or filename not in chunks_by_filename:
            contextual_hits.append(enriched)
            continue

        filename_chunks = chunks_by_filename[filename]
        position = next((i for i, (_, chunk) in enumerate(filename_chunks) if chunk.get("chunk_id") == chunk_id), None)
        if position is None:
            contextual_hits.append(enriched)
            continue

        start = max(0, position - window_size)
        end = min(len(filename_chunks), position + window_size + 1)
        window_chunks = [filename_chunks[i][1].get("text", "").strip() for i in range(start, end)]
        window_chunks = [text for text in window_chunks if text]
        if window_chunks:
            enriched["text"] = "\n\n".join(window_chunks)
        contextual_hits.append(enriched)

    return contextual_hits


class KnowledgeBase:
    def __init__(self):
        self.parser = PDFParser()
        self.chunker = TextChunker(
            max_chars=settings.max_chunk_chars,
            overlap=settings.chunk_overlap,
        )
        self.embedder = Embedder(settings.embedding_model_name)
        self.chunks_path = DATA_DIR / "chunks.json"
        self.index_path = DATA_DIR / "faiss.index"
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.index: faiss.IndexFlatIP | None = None
        self._load()

    def _load(self) -> None:
        if self.chunks_path.exists():
            self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.chunks:
            self._rebuild_bm25()

    def has_data(self) -> bool:
        return bool(self.chunks) and self.index is not None

    def ingest_pdf(self, filename: str, file_bytes: bytes) -> tuple[int, list[str]]:
        warnings: list[str] = []
        pages = self.parser.extract_pages(file_bytes)
        if not pages:
            return 0, [f"{filename}: no extractable text found"]

        new_chunks: list[dict[str, Any]] = []
        for page in pages:
            chunks = self.chunker.chunk_page(page.text)
            if not chunks:
                continue
            for chunk in chunks:
                new_chunks.append(
                    {
                        "chunk_id": str(uuid4())[:8],
                        "filename": filename,
                        "page_number": page.page_number,
                        "text": chunk,
                    }
                )

        if not new_chunks:
            warnings.append(f"{filename}: extracted text was too sparse to chunk")
            return 0, warnings

        embeddings = self.embedder.encode([chunk["text"] for chunk in new_chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self._rebuild_bm25()
        self._persist()
        return len(new_chunks), warnings

    def search(self, query: str) -> list[dict[str, Any]]:
        semantic_hits = self._semantic_search(query, settings.semantic_top_k)
        keyword_hits = self._keyword_search(query, settings.keyword_top_k)
        return fuse_results(
            semantic_hits=semantic_hits,
            keyword_hits=keyword_hits,
            query=query,
            final_top_k=settings.final_top_k,
        )

    def _semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []

        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        hits = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["semantic_score"] = float(score)
            chunk["rank"] = rank
            hits.append(chunk)
        return hits

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.bm25 is None or not self.chunks:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[: min(top_k, len(scores))]

        hits = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["keyword_score"] = float(scores[idx])
            chunk["rank"] = rank
            hits.append(chunk)
        return hits

    def _rebuild_bm25(self) -> None:
        tokenized = [tokenize_for_bm25(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def _persist(self) -> None:
        self.chunks_path.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
