from fastapi import FastAPI

from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.core.config import settings
from app.services.ambiguity import AmbiguityChecker
from app.services.generator import AnswerGenerator
from app.services.query_processor import QueryProcessor
from app.services.storage import KnowledgeBase


app = FastAPI(title=settings.app_name)

app.state.knowledge_base = KnowledgeBase()
app.state.query_processor = QueryProcessor()
app.state.generator = AnswerGenerator(settings.mistral_api_key, settings.mistral_chat_model)
app.state.ambiguity_checker = AmbiguityChecker(settings.mistral_api_key, settings.mistral_chat_model)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "PDF RAG API is running."}
