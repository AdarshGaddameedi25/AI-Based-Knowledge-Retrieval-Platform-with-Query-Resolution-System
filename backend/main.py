from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import ingest, query, health

app = FastAPI(
    title="AI-Based Knowledge Retrieval Platform",
    description="Milestone 1 — RAG Foundation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    uvicorn.run("backend.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
