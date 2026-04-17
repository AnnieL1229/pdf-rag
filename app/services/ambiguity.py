from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.utils.mistral_chat import mistral_chat_messages


@dataclass
class CoverageDecision:
    coverage_sufficient: bool
    needs_clarification: bool
    missing_components: list[str]
    validation_reason: str
    clarification_question: str


def _coerce_bool(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(int(val))
    return None


class AmbiguityChecker:
    def __init__(self, api_key: str | None, chat_model: str = "mistral-small-latest"):
        self.api_key = api_key
        self.chat_model = chat_model

    def detect(
        self,
        query: str,
        rewritten_query: str,
        answer_format: str,
        retrieved_chunks: list[dict],
    ) -> CoverageDecision:
        if not retrieved_chunks:
            return CoverageDecision(
                coverage_sufficient=False,
                needs_clarification=False,
                missing_components=["No retrieved evidence"],
                validation_reason="No retrieved chunks were available for coverage validation.",
                clarification_question="",
            )

        try:
            payload = self._check_with_llm(
                query=query,
                rewritten_query=rewritten_query,
                answer_format=answer_format,
                retrieved_chunks=retrieved_chunks[:5],
            )
        except Exception:
            payload = None

        if not payload:
            return CoverageDecision(
                coverage_sufficient=False,
                needs_clarification=False,
                missing_components=["Coverage validator unavailable"],
                validation_reason=(
                    "Coverage validation failed because the language-model validator was unavailable "
                    "or returned invalid output."
                ),
                clarification_question="",
            )

        coverage_sufficient = bool(payload.get("coverage_sufficient", True))
        needs_clarification = bool(payload.get("needs_clarification", False))
        missing_components = payload.get("missing_components", [])
        if not isinstance(missing_components, list):
            missing_components = []
        missing_components = [str(item).strip() for item in missing_components if str(item).strip()]
        validation_reason = payload.get("reason", "").strip()
        clarification_question = payload.get("clarification_question", "").strip()

        # Keep validation reason concise for UI display.
        if validation_reason:
            validation_reason = re.split(r"(?<=[.?！!])\s+", validation_reason.strip())[0]

        # Keep clarification question short and focused: only the first sentence.
        if clarification_question:
            first_sentence = re.split(r"(?<=[.?！!])\s+", clarification_question.strip())[0]
            clarification_question = first_sentence

        if not coverage_sufficient and needs_clarification and not clarification_question:
            clarification_question = (
                "I found multiple possible answers in the documents. Could you clarify which one you are referring to?"
            )
        return CoverageDecision(
            coverage_sufficient=coverage_sufficient,
            needs_clarification=needs_clarification,
            missing_components=missing_components,
            validation_reason=validation_reason,
            clarification_question=clarification_question,
        )

    def _check_with_llm(
        self,
        query: str,
        rewritten_query: str,
        answer_format: str,
        retrieved_chunks: list[dict],
    ) -> dict | None:
        compact_chunks = [
            {
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", "")[:500],
                "final_score": chunk.get("final_score"),
            }
            for chunk in retrieved_chunks
        ]

        prompt = (
            "You are validating retrieval coverage for a RAG system before answer generation.\n"
            "Decide if retrieved evidence is sufficient to produce a useful, honest answer.\n\n"
            "Return ONLY valid JSON with fields:\n"
            "- coverage_sufficient: boolean\n"
            "- needs_clarification: boolean\n"
            "- missing_components: array of strings\n"
            "- reason: string\n"
            "- clarification_question: string\n\n"
            "Rules:\n"
            "- Identify all entities/subjects and attributes requested by the query.\n"
            "- Check whether retrieved chunks support each required component.\n"
            "- For multi-entity or multi-part questions, track coverage entity-by-entity / part-by-part.\n"
            "- If one or more explicitly requested entities/parts are unsupported, include them in missing_components.\n"
            "- Prefer useful best-effort coverage over strict rejection when evidence is directly relevant to the core question.\n"
            "- For comparisons and multi-part asks, allow partial coverage as sufficient if you can still produce an honest answer with clear limitations.\n"
            "- Do not infer extra required fields that the user did not ask for.\n"
            "- If the evidence clearly supports a high-level but honest answer (and you can state which parts are not covered), set coverage_sufficient=true and needs_clarification=false.\n"
            "- Only set needs_clarification=true when the query is genuinely ambiguous or multiple incompatible answers exist and you cannot decide which matches the user's intent.\n"
            "- For broad asks like contact information or overview questions, partial but directly relevant evidence can still be sufficient unless the user explicitly asks for complete/all details.\n"
            "- If evidence is fundamentally incomplete for a meaningful answer, set coverage_sufficient=false and describe missing_components.\n"
            "- If ambiguity causes incompleteness and a short clarification would resolve it, set needs_clarification=true and provide ONE short clarification_question (no more than ~20 words, no extra explanation).\n"
            "- Keep reason to one short sentence (about 8-20 words).\n"
            "- If coverage_sufficient is true, missing_components must be empty and clarification_question must be empty.\n"
            "- Do not answer the user question itself.\n\n"
            f"Original query: {json.dumps(query)}\n"
            f"Rewritten query: {json.dumps(rewritten_query)}\n"
            f"Requested answer format: {json.dumps(answer_format)}\n"
            f"Retrieved chunks: {json.dumps(compact_chunks)}\n"
        )

        if not self.api_key:
            return None
        raw = mistral_chat_messages(
            self.api_key,
            self.chat_model,
            [{"role": "user", "content": prompt}],
        )
        parsed = self._extract_json_object(raw)
        if not parsed:
            return None

        try:
            data = json.loads(parsed)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None
        cov = _coerce_bool(data.get("coverage_sufficient"))
        clar = _coerce_bool(data.get("needs_clarification"))
        if cov is None or clar is None:
            return None
        data["coverage_sufficient"] = cov
        data["needs_clarification"] = clar
        if not isinstance(data.get("missing_components", []), list):
            return None
        if not isinstance(data.get("clarification_question", ""), str):
            return None
        if not isinstance(data.get("reason", ""), str):
            return None
        return data

    # Heuristic helpers intentionally omitted: decisions should primarily
    # follow the LLM's structured judgment to keep the behavior general and
    # domain-agnostic.

    def _extract_json_object(self, raw: str) -> str | None:
        text = raw.strip()
        if not text:
            return None

        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1].strip()
