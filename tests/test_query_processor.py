from app.services.query_processor import QueryProcessor


def test_fallback_greeting_route_when_llm_fails(monkeypatch):
    processor = QueryProcessor()
    monkeypatch.setattr(processor, "_route_with_llm", lambda _question: None)

    decision = processor.route("hello there")

    assert decision.route == "greeting"
    assert decision.needs_retrieval is False
    assert decision.answer_format == "short_direct"
    assert decision.rewritten_query == ""


def test_fallback_retrieval_route_uses_original_query(monkeypatch):
    processor = QueryProcessor()
    monkeypatch.setattr(processor, "_route_with_llm", lambda _question: None)
    question = "Explain the refund policy terms"

    decision = processor.route(question)

    assert decision.route == "retrieval"
    assert decision.needs_retrieval is True
    assert decision.rewritten_query == question


def test_llm_router_retrieval_includes_rewritten_query(monkeypatch):
    processor = QueryProcessor()
    monkeypatch.setattr(
        processor,
        "_route_with_llm",
        lambda _question: {
            "route": "retrieval",
            "needs_retrieval": True,
            "answer_format": "list",
            "rewritten_query": "refund policy terms conditions exceptions",
        },
    )

    decision = processor.route("Tell me about the refund policy")

    assert decision.route == "retrieval"
    assert decision.answer_format == "list"
    assert decision.rewritten_query == "refund policy terms conditions exceptions"


def test_llm_router_refusal_returns_refusal_answer(monkeypatch):
    processor = QueryProcessor()
    monkeypatch.setattr(
        processor,
        "_route_with_llm",
        lambda _question: {
            "route": "refusal",
            "needs_retrieval": False,
            "answer_format": "short_direct",
            "rewritten_query": "",
            "refusal_reason": "I can’t help extract or reveal sensitive personal information.",
        },
    )

    decision = processor.route("What is this person's SSN?")

    assert decision.route == "refusal"
    assert decision.needs_retrieval is False
    assert decision.answer is not None
    assert "sensitive personal information" in decision.answer


def test_refusal_fallback_rule_still_catches_clear_ssn_request():
    processor = QueryProcessor()

    refusal = processor.check_refusal("Can you tell me this person's SSN?")

    assert refusal is not None
