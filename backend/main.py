import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from backend.api.routes import ingest, query, health, history

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup tasks → yield → (shutdown tasks if any)."""
    logger.info("Starting AI Knowledge Retrieval Platform v2.1 (Milestone 2)")
    logger.info("LLM: %s via OpenRouter", settings.llm_model)
    logger.info("Embedding: %s (local, 384-dim)", settings.embedding_model)
    logger.info("Similarity threshold: %s", settings.similarity_threshold)
    logger.info("Max session history: %s turns", settings.max_session_history)
    try:
        from ingestion.embeddings import EmbeddingService
        EmbeddingService()
        logger.info("Embedding model pre-warmed successfully")
    except Exception as e:
        logger.warning("Could not pre-warm embedding model: %s", e)
    yield
    # Shutdown tasks (none currently needed)


app = FastAPI(
    title="AI-Based Knowledge Retrieval Platform",
    description=(
        "Production RAG system — PostgreSQL + pgvector + OpenRouter + Multi-Agent Orchestration. "
        "Milestone 2: Query Understanding, Retrieval, Response Generation, Orchestration."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ingest.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(history.router, prefix="/api", tags=["History & Stats"])

import os
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
