import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.db.models import Query as QueryModel, Response as ResponseModel
from backend.services.session_store import session_store
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
    Main query endpoint — M2 Multi-Agent RAG pipeline.

    Returns one of:
    - type="direct"        — greeting or small-talk (no retrieval)
    - type="clarification" — ambiguous query, clarification question returned
    - type="rag_response"  — grounded answer with source citations
    - type="error"         — retrieval or LLM failure with controlled message

    Response schema includes M2 canonical fields:
      query_type, routing, detected_intent, detected_domain,
      classification_confidence, confidence, retrieval_count, sources
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
            similarity_threshold=0.0,  # Retriever applies settings.similarity_threshold internally
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

    # Determine response type for the frontend
    if result.clarification_needed:
        response_type = "clarification"
    elif result.retrieval_count > 0 or result.routing == "retrieval":
        response_type = "rag_response"
    else:
        response_type = "direct"

    # Persist to DB for history (only substantive knowledge queries)
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
            logger.warning("Failed to persist query to DB: %s", e)
            db.rollback()

    return {
        # Response classification
        "type": response_type,

        # Query information
        "query": result.query,
        "query_type": result.query_type,              # M2 canonical field
        "routing": result.routing,                    # M2 canonical field
        "detected_intent": result.detected_intent,    # backward compat alias
        "detected_domain": result.detected_domain,
        "key_terms": result.key_terms,

        # Answer
        "answer": result.answer,

        # Source attribution
        "sources": result.sources,

        # Clarification state
        "clarification_needed": result.clarification_needed,
        "clarification_question": result.clarification_question,

        # Scoring
        "confidence": result.confidence,
        "classification_confidence": result.classification_confidence,
        "retrieval_count": result.retrieval_count,

        # Session
        "session_id": session_id,
    }


@router.post("/query/clear-session")
def clear_session(request: ClearSessionRequest):
    """Clear the conversation memory for a given session."""
    cleared = session_store.clear(request.session_id)
    return {"cleared": cleared, "session_id": request.session_id}
