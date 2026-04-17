from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.retriever import evidence_is_strong
from app.services.storage import attach_neighbor_context


router = APIRouter(tags=["query"])


def _best_effort_answer_or_fallback(generated_answer: str) -> str:
    return generated_answer.strip() or "I couldn't fully answer the question from the uploaded documents."


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, payload: QueryRequest) -> QueryResponse:
    query_processor = request.app.state.query_processor
    generator = request.app.state.generator
    knowledge_base = request.app.state.knowledge_base
    ambiguity_checker = request.app.state.ambiguity_checker
    question = payload.question

    decision = query_processor.route(question)
    answer_format = decision.answer_format
    if not decision.needs_retrieval:
        refusal_reason = decision.answer if decision.route == "refusal" else None
        return QueryResponse(
            answer=decision.answer or "",
            rewritten_query=None,
            retrieval_used=False,
            route=decision.route,
            answer_format=answer_format,
            insufficient_evidence=False,
            coverage_sufficient=True,
            needs_clarification=False,
            missing_components=[],
            clarification_question="",
            validation_reason="No retrieval needed for this route.",
            refusal_reason=refusal_reason,
            citations=[],
            retrieved_chunks=[],
        )

    if not knowledge_base.has_data():
        raise HTTPException(
            status_code=400,
            detail="No indexed PDF data is available yet. Upload at least one PDF first.",
        )

    rewritten_query = decision.rewritten_query or question
    hits = knowledge_base.search(rewritten_query)
    if not hits:
        missing_components = ["No relevant retrieved context"]
        return QueryResponse(
            answer="I couldn't find enough relevant context in the indexed PDFs to answer confidently.",
            rewritten_query=rewritten_query,
            retrieval_used=True,
            route=decision.route,
            answer_format=answer_format,
            insufficient_evidence=True,
            coverage_sufficient=False,
            needs_clarification=False,
            missing_components=missing_components,
            clarification_question="",
            validation_reason="Retrieval returned no chunks.",
            refusal_reason=None,
            citations=[],
            retrieved_chunks=[],
        )

    if not evidence_is_strong(hits):
        missing_components = ["Retrieved evidence scores were below confidence threshold"]
        return QueryResponse(
            answer="I couldn't verify enough evidence in the uploaded documents to answer confidently.",
            rewritten_query=rewritten_query,
            retrieval_used=True,
            route=decision.route,
            answer_format=answer_format,
            insufficient_evidence=True,
            coverage_sufficient=False,
            needs_clarification=False,
            missing_components=missing_components,
            clarification_question="",
            validation_reason="Evidence threshold gate failed.",
            refusal_reason=None,
            citations=[],
            retrieved_chunks=[SourceChunk(**hit) for hit in hits],
        )

    llm_hits = attach_neighbor_context(hits, knowledge_base.chunks, window_size=1)

    coverage = ambiguity_checker.detect(
        query=question,
        rewritten_query=rewritten_query,
        answer_format=answer_format or "default_explanatory",
        retrieved_chunks=llm_hits,
    )
    if not coverage.coverage_sufficient:
        source_chunks = [SourceChunk(**hit) for hit in hits]
        if coverage.needs_clarification:
            return QueryResponse(
                answer="",
                rewritten_query=rewritten_query,
                retrieval_used=True,
                route=decision.route,
                answer_format=answer_format,
                insufficient_evidence=False,
                coverage_sufficient=False,
                needs_clarification=True,
                missing_components=coverage.missing_components,
                clarification_question=coverage.clarification_question,
                validation_reason=coverage.validation_reason,
                refusal_reason=None,
                citations=source_chunks,
                retrieved_chunks=source_chunks,
            )

        try:
            partial_answer = generator.answer(
                question,
                llm_hits,
                answer_format=answer_format,
                require_partial_mode=True,
            )
        except Exception:
            partial_answer = ""
        return QueryResponse(
            answer=_best_effort_answer_or_fallback(partial_answer),
            rewritten_query=rewritten_query,
            retrieval_used=True,
            route=decision.route,
            answer_format=answer_format,
            insufficient_evidence=False,
            coverage_sufficient=False,
            needs_clarification=False,
            missing_components=coverage.missing_components,
            clarification_question="",
            validation_reason=coverage.validation_reason,
            refusal_reason=None,
            citations=source_chunks,
            retrieved_chunks=source_chunks,
        )

    try:
        answer = generator.answer(question, llm_hits, answer_format=answer_format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {exc}") from exc

    source_chunks = [SourceChunk(**hit) for hit in hits]
    return QueryResponse(
        answer=answer,
        rewritten_query=rewritten_query,
        retrieval_used=True,
        route=decision.route,
        answer_format=answer_format,
        insufficient_evidence=False,
        coverage_sufficient=True,
        needs_clarification=False,
        missing_components=[],
        clarification_question="",
        validation_reason=coverage.validation_reason or "Coverage validation passed.",
        refusal_reason=None,
        citations=source_chunks,
        retrieved_chunks=source_chunks,
    )
