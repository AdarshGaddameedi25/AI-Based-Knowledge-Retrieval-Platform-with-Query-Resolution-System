import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.db.models import Query as QueryModel, RetrievalResult as RetrievalResultModel, Response as ResponseModel
from retrieval.rag_pipeline import RAGPipeline
from agents.query_understanding_agent import QueryUnderstandingAgent

router = APIRouter()
logger = logging.getLogger(__name__)

_query_agent = QueryUnderstandingAgent()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = Field(default=None)


@router.post("/query")
def handle_query(request: QueryRequest, db: Session = Depends(get_db)):
    query_text = request.query.strip()

    analysis = _query_agent.analyze(query_text)

    if not analysis.requires_retrieval:
        return {
            "type": "direct_response",
            "query": query_text,
            "answer": "Hello! I am the AI Knowledge Retrieval Assistant. Ask me anything about the uploaded documents.",
            "sources": [],
            "retrieval_results": [],
        }

    try:
        pipeline = RAGPipeline()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    db_query = QueryModel(query_text=query_text, session_id=request.session_id)
    db.add(db_query)
    db.flush()

    try:
        rag_response = pipeline.run(query_text, db=db, top_k=request.top_k)
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    for r in rag_response.retrieval_results:
        db_result = RetrievalResultModel(
            query_id=db_query.query_id,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            similarity_score=r.similarity_score,
            rank=r.rank,
        )
        db.add(db_result)

    db_response = ResponseModel(
        query_id=db_query.query_id,
        answer=rag_response.answer,
    )
    db.add(db_response)
    db.commit()

    return {
        "type": "rag_response",
        "query": rag_response.query,
        "answer": rag_response.answer,
        "sources": rag_response.sources,
        "detected_intent": analysis.detected_intent,
        "key_terms": analysis.key_terms,
        "retrieval_count": len(rag_response.retrieval_results),
    }
