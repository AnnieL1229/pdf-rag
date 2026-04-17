from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import IngestResponse


router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    request: Request,
    files: list[UploadFile] = File(...),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    knowledge_base = request.app.state.knowledge_base
    processed = 0
    total_chunks = 0
    filenames: list[str] = []
    skipped_files: list[str] = []
    warnings: list[str] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            skipped_files.append(file.filename)
            warnings.append(f"{file.filename}: skipped because it is not a PDF")
            continue

        file_bytes = await file.read()
        if not file_bytes:
            skipped_files.append(file.filename)
            warnings.append(f"{file.filename}: file was empty")
            continue

        chunk_count, file_warnings = knowledge_base.ingest_pdf(file.filename, file_bytes)
        warnings.extend(file_warnings)
        if chunk_count == 0:
            skipped_files.append(file.filename)
            continue

        processed += 1
        total_chunks += chunk_count
        filenames.append(file.filename)

    return IngestResponse(
        files_processed=processed,
        chunks_created=total_chunks,
        filenames=filenames,
        skipped_files=skipped_files,
        warnings=warnings,
    )
