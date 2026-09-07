import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.db.models import Query as QueryModel, RetrievalResult as RetrievalResultModel, Response as ResponseModel
from backend.services.session_store import session_store
from agents.orchestrator import AgentOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

# Singleton orchestrator — shared across requests (all internal state is per-call)
_orchestrator = AgentOrchestrator()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = Field(default=None, description="Pass an existing session_id to continue a conversation")
    domain_filter: Optional[str] = Field(default=None, description="Restrict retrieval to a specific domain (hr, technology, legal)")


class ClearSessionRequest(BaseModel):
    session_id: str


@router.post("/query")
def handle_query(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Main query endpoint.

    Returns:
    - Direct answer for greetings / small talk
    - Clarification question if query is ambiguous
    - RAG answer with sources, domain, and confidence for knowledge queries
    """
    query_text = request.query.strip()

    # Retrieve or create session memory
    session_id, memory = session_store.get_or_create(request.session_id)

    try:
        result = _orchestrator.run(
            query=query_text,
            db=db,
            memory=memory,
            session_id=session_id,
            top_k=request.top_k,
            domain_filter=request.domain_filter,
            similarity_threshold=0.0,  # Retriever uses settings.similarity_threshold by default
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Orchestrator error for query '{query_text[:80]}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    # Persist to DB (only for knowledge queries that went through retrieval)
    if result.retrieval_count > 0 or (not result.clarification_needed and result.confidence > 0):
        try:
            db_query = QueryModel(query_text=query_text, session_id=session_id)
            db.add(db_query)
            db.flush()

            db_response = ResponseModel(
                query_id=db_query.query_id,
                answer=result.answer,
            )
            db.add(db_response)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist query to DB: {e}")
            db.rollback()

    return {
        "type": "clarification" if result.clarification_needed else "rag_response",
        "query": result.query,
        "answer": result.answer,
        "sources": result.sources,
        "detected_intent": result.detected_intent,
        "detected_domain": result.detected_domain,
        "key_terms": result.key_terms,
        "clarification_needed": result.clarification_needed,
        "clarification_question": result.clarification_question,
        "confidence": result.confidence,
        "retrieval_count": result.retrieval_count,
        "session_id": session_id,
    }


@router.post("/query/clear-session")
def clear_session(request: ClearSessionRequest):
    """Clear the conversation memory for a given session."""
    cleared = session_store.clear(request.session_id)
    return {"cleared": cleared, "session_id": request.session_id}
