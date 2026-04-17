from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    text: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None


class IngestResponse(BaseModel):
    files_processed: int
    chunks_created: int
    filenames: list[str]
    skipped_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    rewritten_query: str | None = None
    retrieval_used: bool
    route: str
    answer_format: str | None = None
    insufficient_evidence: bool = False
    coverage_sufficient: bool = True
    needs_clarification: bool = False
    missing_components: list[str] = Field(default_factory=list)
    clarification_question: str = ""
    validation_reason: str = ""
    refusal_reason: str | None = None
    citations: list[SourceChunk] = Field(default_factory=list)
    retrieved_chunks: list[SourceChunk] = Field(default_factory=list)
