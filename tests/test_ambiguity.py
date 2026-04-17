from app.services.ambiguity import AmbiguityChecker


def test_ambiguity_checker_uses_safe_default_when_llm_fails(monkeypatch):
    checker = AmbiguityChecker(api_key=None)
    monkeypatch.setattr(checker, "_check_with_llm", lambda *_args, **_kwargs: None)

    decision = checker.detect(
        query="What is the deadline?",
        rewritten_query="deadline due date",
        answer_format="short_direct",
        retrieved_chunks=[{"chunk_id": "a", "text": "Item A due March 10"}],
    )

    assert decision.coverage_sufficient is False
    assert decision.needs_clarification is False
    assert "Coverage validator unavailable" in decision.missing_components


def test_ambiguity_checker_extracts_json_from_markdown_block():
    checker = AmbiguityChecker(api_key=None)
    raw = """```json
{"coverage_sufficient": false, "needs_clarification": true, "missing_components": ["target"], "reason": "missing target", "clarification_question": "Which one?"}
```"""

    parsed = checker._extract_json_object(raw)

    assert parsed is not None
    assert parsed.startswith("{")
    assert parsed.endswith("}")


def test_ambiguity_checker_returns_clarification_from_llm(monkeypatch):
    checker = AmbiguityChecker(api_key=None)
    monkeypatch.setattr(
        checker,
        "_check_with_llm",
        lambda *_args, **_kwargs: {
            "coverage_sufficient": False,
            "needs_clarification": True,
            "missing_components": ["specific target item"],
            "reason": "query does not identify one target among multiple candidates",
            "clarification_question": "I found multiple possible answers. Could you clarify?",
        },
    )

    decision = checker.detect(
        query="What is the deadline?",
        rewritten_query="deadline due date",
        answer_format="short_direct",
        retrieved_chunks=[
            {"chunk_id": "a", "text": "Item A due March 10"},
            {"chunk_id": "b", "text": "Item B due April 5"},
        ],
    )

    assert decision.coverage_sufficient is False
    assert decision.needs_clarification is True
    assert "specific target item" in decision.missing_components
    assert "multiple" in decision.clarification_question.lower()
    assert "multiple" in decision.validation_reason
