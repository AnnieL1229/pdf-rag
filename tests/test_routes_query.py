from app.api.routes_query import _best_effort_answer_or_fallback


def test_best_effort_answer_or_fallback_returns_generated_answer():
    answer = _best_effort_answer_or_fallback("Class A allows AI with disclosure.")

    assert answer == "Class A allows AI with disclosure."


def test_best_effort_answer_or_fallback_uses_default_when_empty():
    answer = _best_effort_answer_or_fallback("   ")

    assert answer == "I couldn't fully answer the question from the uploaded documents."
