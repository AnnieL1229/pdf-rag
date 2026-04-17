from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.config import settings
from app.utils.mistral_chat import mistral_chat_messages


CHITCHAT_PATTERNS = {
    "greeting": {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    },
    "gratitude": {"thanks", "thank you", "thx"},
}


@dataclass
class QueryDecision:
    route: str
    needs_retrieval: bool
    answer_format: str
    rewritten_query: str
    answer: str | None = None


FALLBACK_REFUSAL_PATTERNS = {
    "pii": [r"\bssn\b", r"social security number", r"passport number", r"credit card number"],
    "medical": [r"\bmedical advice\b", r"\bdiagnos(?:e|is)\b", r"\btreatment plan\b"],
    "legal": [r"\blegal advice\b", r"\blawsuit\b", r"\bsue\b"],
}


class QueryProcessor:
    def route(self, question: str) -> QueryDecision:
        """
        Route the query and, when needed, produce a retrieval-friendly rewrite.
        Prefer a small LLM router, with a simple rule-based fallback.
        """
        try:
            result = self._route_with_llm(question)
        except Exception:
            result = None

        if not result:
            return self._route_with_fallback(question)

        route = result.get("route", "retrieval")
        needs_retrieval = bool(result.get("needs_retrieval", route == "retrieval"))
        answer_format = result.get("answer_format") or self.classify_answer_format(question)
        rewritten_query = result.get("rewritten_query") or ""

        # If LLM says no retrieval, ignore any rewritten text
        if not needs_retrieval:
            rewritten_query = ""

        # Map refusal route to a fixed message (LLM does not answer itself)
        if route == "refusal" and not needs_retrieval:
            message = result.get("refusal_reason") or self.check_refusal(question) or "I can’t help with that request."
            return QueryDecision(
                route="refusal",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer=message,
            )

        if route == "greeting" and not needs_retrieval:
            return QueryDecision(
                route="greeting",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer="Hello. Upload one or more PDFs and ask a question about them when you're ready.",
            )

        if route == "gratitude" and not needs_retrieval:
            return QueryDecision(
                route="gratitude",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer="You're welcome.",
            )

        if route == "help" and not needs_retrieval:
            return QueryDecision(
                route="help",
                needs_retrieval=False,
                answer_format="default_explanatory",
                rewritten_query="",
                answer=(
                    "I can ingest PDF files and answer questions grounded in the uploaded documents."
                ),
            )

        # Default: document-style question that should go through retrieval.
        if needs_retrieval and not rewritten_query:
            # If the model did not give us anything helpful, fall back to the raw question.
            rewritten_query = question.strip()

        return QueryDecision(
            route="retrieval",
            needs_retrieval=True,
            answer_format=answer_format,
            rewritten_query=rewritten_query,
            answer=None,
        )

    def classify_answer_format(self, question: str) -> str:
        lowered = question.strip().lower()

        if any(word in lowered for word in ["list", "bullet", "items", "requirements"]):
            return "list"
        if any(word in lowered for word in ["compare", "difference between", "versus", "vs."]):
            return "table"
        if any(
            lowered.startswith(prefix)
            for prefix in ["what is ", "what's ", "what are ", "define ", "how long "]
        ):
            return "short_direct"
        return "default_explanatory"

    def check_refusal(self, question: str) -> str | None:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        for category, patterns in FALLBACK_REFUSAL_PATTERNS.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                if category == "pii":
                    return "I can’t help extract or reveal sensitive personal information."
                if category == "medical":
                    return "I can’t provide medical advice. Please consult a qualified medical professional."
                if category == "legal":
                    return "I can’t provide legal advice. Please consult a qualified legal professional."
        return None

    def _route_with_llm(self, question: str) -> dict | None:
        """
        Ask a small model to classify the query into route/needs_retrieval/answer_format
        and optionally produce a retrieval-friendly rewritten_query.
        Returns a dict or None on failure.
        """
        prompt = (
            "You are a small router for a PDF QA system.\n"
            "Given a user query, decide how it should be handled.\n\n"
            "Always return ONLY a JSON object with these fields:\n"
            "- route: one of [greeting, gratitude, help, retrieval, refusal]\n"
            "- needs_retrieval: true or false\n"
            "- answer_format: one of [short_direct, list, table, default_explanatory]\n"
            "- rewritten_query: a short retrieval-oriented query, or an empty string if needs_retrieval is false\n\n"
            "Optional fields:\n"
            "- refusal_category: one of [pii, medical, legal, none]\n"
            "- refusal_reason: short message only when route is refusal\n\n"
            "Routing rules:\n"
            "- greeting: hi/hello/hey/good morning, needs_retrieval=false, short_direct, rewritten_query=\"\"\n"
            "- gratitude: thanks/thank you, needs_retrieval=false, short_direct, rewritten_query=\"\"\n"
            "- help: questions like 'help' or 'what can you do', needs_retrieval=false, default_explanatory, rewritten_query=\"\"\n"
            "- refusal: sensitive PII, medical advice, or legal advice, needs_retrieval=false, short_direct, rewritten_query=\"\"\n"
            "- retrieval: default for document questions, needs_retrieval=true and provide a retrieval-friendly rewritten_query\n\n"
            "Policy nuance:\n"
            "- Do not mark as refusal for normal document lookup of public information (for example, course syllabus contact details).\n"
            "- If unsure, prefer retrieval rather than refusal.\n\n"
            "Rewriting rules:\n"
            "- Only rewrite when needs_retrieval is true.\n"
            "- rewritten_query should be a short bag of key terms for retrieval, not a full sentence.\n"
            "- Preserve explicit scope constraints and quantifiers from the user query (for example: all, each, every, any, complete, entire).\n"
            "- Preserve requested entity names exactly when possible (course names, product names, policy names, proper nouns).\n"
            "- Do not generalize or drop entities (for example, keep 'NLP class' and do not rewrite it as only 'NLP').\n"
            "- For multi-entity asks, include all requested entities in rewritten_query.\n"
            "- Do not answer the question.\n"
            "- Do not include filler like 'find the section that explains'.\n\n"
            "Examples:\n"
            "Query: \"hi there\"\n"
            "{ \"route\": \"greeting\", \"needs_retrieval\": false, \"answer_format\": \"short_direct\", \"rewritten_query\": \"\" }\n\n"
            "Query: \"thanks a lot\"\n"
            "{ \"route\": \"gratitude\", \"needs_retrieval\": false, \"answer_format\": \"short_direct\", \"rewritten_query\": \"\" }\n\n"
            "Query: \"what can you do\"\n"
            "{ \"route\": \"help\", \"needs_retrieval\": false, \"answer_format\": \"default_explanatory\", \"rewritten_query\": \"\" }\n\n"
            "Query: \"What is the cancellation policy?\"\n"
            "{ \"route\": \"retrieval\", \"needs_retrieval\": true, \"answer_format\": \"default_explanatory\", \"rewritten_query\": \"cancellation policy terms conditions notice period\" }\n\n"
            "Query: \"What are the requirements for cancellation?\"\n"
            "{ \"route\": \"retrieval\", \"needs_retrieval\": true, \"answer_format\": \"list\", \"rewritten_query\": \"cancellation requirements conditions notice steps\" }\n\n"
            "Query: \"Compare refund policy and termination policy\"\n"
            "{ \"route\": \"retrieval\", \"needs_retrieval\": true, \"answer_format\": \"table\", \"rewritten_query\": \"refund policy termination policy comparison fees notice exceptions\" }\n\n"
            "Query: \"List professor names for all classes\"\n"
            "{ \"route\": \"retrieval\", \"needs_retrieval\": true, \"answer_format\": \"list\", \"rewritten_query\": \"all classes professor names instructor names\" }\n\n"
            "Query: \"List the grading of Power and Negotiation class and NLP class\"\n"
            "{ \"route\": \"retrieval\", \"needs_retrieval\": true, \"answer_format\": \"list\", \"rewritten_query\": \"Power and Negotiation class grading NLP class grading\" }\n\n"
            "Query: \"What is this person's SSN?\"\n"
            "{ \"route\": \"refusal\", \"needs_retrieval\": false, \"answer_format\": \"short_direct\", \"rewritten_query\": \"\" }\n\n"
            f"Query: {json.dumps(question)}\n"
        )

        if not settings.mistral_api_key:
            return None
        raw = mistral_chat_messages(
            settings.mistral_api_key,
            settings.mistral_chat_model,
            [{"role": "user", "content": prompt}],
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        route = data.get("route")
        needs_retrieval = data.get("needs_retrieval")
        answer_format = data.get("answer_format")
        rewritten_query = data.get("rewritten_query", "")

        if route not in {"greeting", "gratitude", "help", "retrieval", "refusal"}:
            return None
        if not isinstance(needs_retrieval, bool):
            return None
        if answer_format not in {
            "short_direct",
            "list",
            "table",
            "default_explanatory",
        }:
            data["answer_format"] = self.classify_answer_format(question)
        if not isinstance(rewritten_query, str):
            data["rewritten_query"] = ""
        if "refusal_reason" in data and not isinstance(data["refusal_reason"], str):
            data["refusal_reason"] = ""

        return data

    def _route_with_fallback(self, question: str) -> QueryDecision:
        lowered = question.strip().lower()
        normalized = re.sub(r"[^\w\s]", "", lowered)

        # Simple refusal guard
        refusal = self.check_refusal(question)
        if refusal:
            return QueryDecision(
                route="refusal",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer=refusal,
            )

        if any(greet in normalized for greet in ["hello", "hi", "hey", "good morning"]):
            return QueryDecision(
                route="greeting",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer="Hello. Upload one or more PDFs and ask a question about them when you're ready.",
            )

        if any(thanks in normalized for thanks in ["thanks", "thank you", "thx"]):
            return QueryDecision(
                route="gratitude",
                needs_retrieval=False,
                answer_format="short_direct",
                rewritten_query="",
                answer="You're welcome.",
            )

        if "help" in normalized or "what can you do" in normalized or "how does this work" in normalized:
            return QueryDecision(
                route="help",
                needs_retrieval=False,
                answer_format="default_explanatory",
                rewritten_query="",
                answer=(
                    "I can ingest PDF files and answer questions grounded in the uploaded documents."
                ),
            )

        # Fallback retrieval: no rewrite, just use the raw question downstream.
        return QueryDecision(
            route="retrieval",
            needs_retrieval=True,
            answer_format=self.classify_answer_format(question),
            rewritten_query=question.strip(),
            answer=None,
        )
