from __future__ import annotations

import re

from app.utils.mistral_chat import mistral_chat_messages
from app.utils.text import keyword_overlap_score, tokenize_for_bm25


class AnswerGenerator:
    def __init__(self, api_key: str | None, chat_model: str = "mistral-small-latest"):
        self.api_key = api_key
        self.chat_model = chat_model

    def build_context(self, hits: list[dict]) -> str:
        sections = []
        for hit in hits:
            sections.append(
                f"[{hit['filename']} - page {hit['page_number']} - {hit['chunk_id']}]\n{hit['text']}"
            )
        return "\n\n".join(sections)

    def _format_instruction(self, answer_format: str | None) -> str:
        if answer_format == "list":
            return (
                "Respond with a short list of bullet points that directly answer the question, "
                "grounded only in the context."
            )
        if answer_format == "table":
            return (
                "If the context supports it, answer using a small markdown table that compares the "
                "relevant items. Otherwise fall back to a short paragraph."
            )
        if answer_format == "short_direct":
            return (
                "Give a concise direct answer in 1–2 sentences, followed by a brief mention of the "
                "source filename and page number where applicable."
            )
        return (
            "Respond with a short explanatory answer (one or two short paragraphs) grounded only "
            "in the context."
        )

    def _clean_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def filter_by_evidence(self, answer: str, hits: list[dict]) -> str:
        if not answer.strip() or not hits:
            return answer

        context_text = "\n\n".join(hit["text"] for hit in hits)
        answer_sentences = self._clean_sentences(answer)
        if not answer_sentences:
            return answer

        kept: list[str] = []
        unsupported_count = 0
        for sentence in answer_sentences:
            terms = tokenize_for_bm25(sentence)
            score = keyword_overlap_score(terms, context_text)
            if score >= 0.15:
                kept.append(sentence)
            else:
                unsupported_count += 1

        if not kept:
            return "I could not verify enough evidence in the retrieved documents to answer confidently."

        if unsupported_count > len(answer_sentences) / 2:
            return "I could not verify enough evidence in the retrieved documents to answer confidently."

        return " ".join(kept)

    def answer(
        self,
        question: str,
        hits: list[dict],
        answer_format: str | None,
        require_partial_mode: bool = False,
    ) -> str:
        context = self.build_context(hits)
        format_instruction = self._format_instruction(answer_format)
        partial_instruction = (
            "Return a best-effort partial answer using only supported evidence."
            if require_partial_mode
            else "If evidence is incomplete, avoid guessing and keep unsupported parts explicit."
        )
        prompt = (
            "Use only the context below to answer the user's question.\n"
            "If the answer is not supported by the context, say you do not know.\n"
            "If the retrieved context does not fully support all parts of the question, do not guess or fill gaps.\n"
            "If the question is ambiguous or could refer to multiple items in the context, do not guess. Ask the user to clarify.\n"
            "For multi-entity or multi-part questions, include only entities/parts directly supported by context evidence.\n"
            "Do not invent details for unsupported entities or missing sub-questions.\n"
            f"{partial_instruction}\n"
            "When possible, mention the source filename and page number.\n"
            f"{format_instruction}\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )

        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set. Add it to your .env file.")
        raw_answer = mistral_chat_messages(
            self.api_key,
            self.chat_model,
            [
                {
                    "role": "system",
                    "content": "You answer questions using provided document context only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return self.filter_by_evidence(raw_answer, hits)
