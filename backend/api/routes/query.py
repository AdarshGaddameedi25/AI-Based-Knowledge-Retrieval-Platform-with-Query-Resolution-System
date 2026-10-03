import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.db.models import Query as QueryModel, Response as ResponseModel
from backend.services.session_store import session_store
from backend.services.analytics_service import analytics_service
from agents.orchestrator import AgentOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

# Singleton orchestrator — shared across requests (all internal state is per-call)
_orchestrator = AgentOrchestrator()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = Field(
        default=None,
        description="Pass an existing session_id to continue a conversation",
    )
    domain_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a specific domain (hr, technology, legal)",
    )


class ClearSessionRequest(BaseModel):
    session_id: str


@router.post("/query")
def handle_query(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Main query endpoint — M3 Multi-Agent RAG pipeline + M4 Analytics.

    Returns one of:
    - type="direct"        — greeting or small-talk (no retrieval)
    - type="clarification" — ambiguous query, clarification question returned
    - type="rag_response"  — grounded answer with source citations
    - type="error"         — retrieval or LLM failure with controlled message

    M4 additions:
      Analytics recorded for every query (best-effort).
      Knowledge gaps detected and stored when retrieval fails.
    """
    query_text = request.query.strip()

    # Retrieve or create session memory
    session_id, memory = session_store.get_or_create(request.session_id)

    t_start = time.perf_counter()
    try:
        result = _orchestrator.run(
            query=query_text,
            db=db,
            memory=memory,
            session_id=session_id,
            top_k=request.top_k,
            domain_filter=request.domain_filter,
            similarity_threshold=None,  # None → retriever uses settings.similarity_threshold
        )
    except EnvironmentError as e:
        # Raised when OPENROUTER_API_KEY is missing — configuration error
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            "Orchestrator unhandled error for query %r: %s",
            query_text[:80], e, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )
    latency_ms = (time.perf_counter() - t_start) * 1000

    # Determine response type for the frontend
    if result.clarification_needed:
        response_type = "clarification"
    elif result.retrieval_count > 0 or result.routing == "retrieval":
        response_type = "rag_response"
    else:
        response_type = "direct"

    # Determine analytics response_status
    _GAP_PHRASE = "could not find sufficient information"
    if result.clarification_needed:
        resp_status = "clarification"
    elif result.retrieval_count == 0 and result.routing == "retrieval":
        resp_status = "knowledge_gap"
    elif _GAP_PHRASE in (result.answer or "").lower():
        resp_status = "knowledge_gap"
    else:
        resp_status = "answered"

    # Extract similarity scores from sources
    similarity_scores = [s.get("similarity_score", 0) for s in result.sources] if result.sources else []

    # Persist to DB for history (only substantive knowledge queries)
    persisted_query_id = None
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
            persisted_query_id = db_query.query_id
        except Exception as e:
            logger.warning("Failed to persist query to DB: %s", e)
            db.rollback()

    # M4.1 — Record analytics (best-effort, separate DB operation)
    try:
        analytics_service.record_query(
            db=db,
            original_query=query_text,
            normalized_query=getattr(result, "query", query_text),
            query_type=result.query_type,
            detected_intent=result.detected_intent,
            detected_domain=result.detected_domain,
            routing_path=result.routing,
            clarification_required=result.clarification_needed,
            retrieval_count=result.retrieval_count,
            retrieved_chunks=result.retrieval_count,
            similarity_scores=similarity_scores,
            source_count=len(result.sources),
            response_status=resp_status,
            response_generated=(not result.clarification_needed and resp_status != "error"),
            response_latency_ms=round(latency_ms, 1),
            key_terms=result.key_terms,
            session_id=session_id,
            query_id=persisted_query_id,
        )
    except Exception as e:
        logger.warning("Analytics recording failed (non-fatal): %s", e)

    return {
        # Response classification
        "type": response_type,

        # Query information
        "query": result.query,
        "refined_query": result.refined_query,          # M3.1 — refined after clarification
        "query_type": result.query_type,                # M2 canonical field
        "routing": result.routing,                       # M2 canonical field
        "detected_intent": result.detected_intent,       # backward compat alias
        "detected_domain": result.detected_domain,
        "key_terms": result.key_terms,

        # Answer
        "answer": result.answer,

        # Source attribution — M3.4 includes text + citation
        "sources": result.sources,

        # M3.1 — Structured clarification state
        "clarification": {
            "required": result.clarification_needed,
            "question": result.clarification_question if result.clarification_needed else None,
            "pending": result.clarification_needed,
        },

        # Backward compat flat fields (keep for existing frontend code)
        "clarification_needed": result.clarification_needed,
        "clarification_question": result.clarification_question,

        # M3.2 — Memory state
        "memory": {
            "session_id": session_id,
            "used_context": result.used_memory_context,
            "active_topic": memory.active_topic,
        },

        # Scoring
        "confidence": result.confidence,
        "classification_confidence": result.classification_confidence,
        "retrieval_count": result.retrieval_count,

        # Session (backward compat)
        "session_id": session_id,
    }


@router.post("/query/clear-session")
def clear_session(request: ClearSessionRequest):
    """Clear the conversation memory for a given session."""
    cleared = session_store.clear(request.session_id)
    return {"cleared": cleared, "session_id": request.session_id}

