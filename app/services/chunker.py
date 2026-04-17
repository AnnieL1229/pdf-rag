import re


class TextChunker:
    def __init__(self, max_chars: int = 900, overlap: int = 150):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk_page(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            units = self._split_long_paragraph(paragraph)
            for unit in units:
                candidate = unit if not current else f"{current}\n\n{unit}"
                if len(candidate) <= self.max_chars:
                    current = candidate
                    continue

                if current:
                    chunks.append(current.strip())
                    overlap_text = self._overlap_tail(current)
                else:
                    overlap_text = ""
                if len(unit) <= self.max_chars:
                    combined = unit if not overlap_text else f"{overlap_text}\n\n{unit}"
                    current = combined if len(combined) <= self.max_chars else unit
                else:
                    chunks.extend(self._hard_split(unit))
                    current = ""

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.max_chars:
            return [paragraph]

        pieces = re.split(r"(?<=[.!?])\s+", paragraph)
        if len(pieces) == 1:
            return self._hard_split(paragraph)

        parts: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else f"{current} {piece}"
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                if current:
                    parts.append(current.strip())
                current = piece
        if current:
            parts.append(current.strip())
        return parts

    def _hard_split(self, text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        step = max(1, self.max_chars - self.overlap)
        while start < len(text):
            end = start + self.max_chars
            parts.append(text[start:end].strip())
            start += step
        return [part for part in parts if part]

    def _overlap_tail(self, text: str) -> str:
        if self.overlap <= 0:
            return ""
        return text[-self.overlap :].strip()
