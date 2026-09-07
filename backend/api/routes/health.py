import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.db.base import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "unreachable"
    pgvector_status = "not_enabled"
    document_count = 0
    chunk_count = 0

    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"

        result = db.execute(text("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'"))
        row = result.fetchone()
        if row and row[0]:
            pgvector_status = f"enabled (v{row[0]})"
        else:
            pgvector_status = "available_not_enabled"

        doc_result = db.execute(text("SELECT COUNT(*) FROM documents WHERE status = 'indexed'"))
        document_count = doc_result.scalar()

        chunk_result = db.execute(text("SELECT COUNT(*) FROM embeddings"))
        chunk_count = chunk_result.scalar()

    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        db_status = f"error: {str(e)}"

    return {
        "status": "ok" if db_status == "connected" else "degraded",
        "database": db_status,
        "pgvector": pgvector_status,
        "indexed_documents": document_count,
        "total_chunks": chunk_count,
        "service": "AI Knowledge Retrieval Platform",
    }
