from dataclasses import dataclass

import fitz

from app.utils.text import clean_text


@dataclass
class PageText:
    page_number: int
    text: str


class PDFParser:
    def extract_pages(self, file_bytes: bytes) -> list[PageText]:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[PageText] = []

        for page_index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            pages.append(PageText(page_number=page_index, text=text))

        return pages
