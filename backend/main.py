import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from backend.api.routes import ingest, query, health

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Based Knowledge Retrieval Platform",
    description="Production RAG system — PostgreSQL + pgvector + OpenRouter",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ingest.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query"])

import os
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("startup")
async def startup():
    logger.info("Starting AI Knowledge Retrieval Platform v2.0")
    logger.info(f"LLM: {settings.llm_model} via OpenRouter")
    logger.info(f"Embedding: {settings.embedding_model} (local)")
    try:
        from ingestion.embeddings import EmbeddingService
        EmbeddingService()
        logger.info("Embedding model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Could not pre-warm embedding model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
