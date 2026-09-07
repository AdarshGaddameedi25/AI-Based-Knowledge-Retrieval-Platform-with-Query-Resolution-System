import logging
from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.db.base import get_db
from backend.db.models import Query as QueryModel, Response as ResponseModel, Document, DocumentChunk, Embedding
from retrieval.vector_store import VectorStore

router = APIRouter()
logger = logging.getLogger(__name__)

_vector_store = VectorStore()


@router.get("/history")
def get_query_history(
    limit: int = QueryParam(default=20, ge=1, le=100),
    offset: int = QueryParam(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Paginated query history with answers.
    Returns queries in reverse-chronological order.
    """
    try:
        total = db.query(func.count(QueryModel.query_id)).scalar()
        rows = (
            db.query(QueryModel, ResponseModel)
            .outerjoin(ResponseModel, QueryModel.query_id == ResponseModel.query_id)
            .order_by(QueryModel.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        history = []
        for q, r in rows:
            history.append({
                "query_id": q.query_id,
                "query_text": q.query_text,
                "session_id": q.session_id,
                "timestamp": q.timestamp.isoformat(),
                "answer": r.answer if r else None,
                "answer_timestamp": r.timestamp.isoformat() if r else None,
            })

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "history": history,
        }
    except Exception as e:
        logger.error(f"Error fetching query history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    System statistics: document counts, chunk counts, query counts, domain breakdown.
    """
    try:
        total_documents = db.query(func.count(Document.document_id)).filter(Document.status == "indexed").scalar()
        total_chunks = _vector_store.count_chunks(db)
        total_queries = db.query(func.count(QueryModel.query_id)).scalar()
        domain_stats = _vector_store.get_domain_stats(db)

        # Document status breakdown
        status_rows = (
            db.query(Document.status, func.count(Document.document_id).label("count"))
            .group_by(Document.status)
            .all()
        )
        status_breakdown = {row.status: row.count for row in status_rows}

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_queries": total_queries,
            "domain_breakdown": domain_stats,
            "document_status": status_breakdown,
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
