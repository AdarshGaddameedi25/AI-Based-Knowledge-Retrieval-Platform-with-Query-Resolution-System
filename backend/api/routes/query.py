from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.clarification_agent import ClarificationAgent
from retrieval.rag_pipeline import RAGPipeline

router = APIRouter()

query_agent = QueryUnderstandingAgent()
clarification_agent = ClarificationAgent()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/query")
def handle_query(request: QueryRequest):
    analysis = query_agent.analyze(request.query)

    clarification = clarification_agent.evaluate(request.query)
    if clarification.is_ambiguous:
        return {
            "type": "clarification_needed",
            "clarification_question": clarification.clarification_question,
            "original_query": request.query,
        }

    if not analysis.requires_retrieval:
        return {
            "type": "direct_response",
            "answer": "Hello! I am the AI Knowledge Retrieval Assistant. How can I help you?",
            "query": request.query,
        }

    try:
        pipeline = RAGPipeline()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = pipeline.run(request.query, top_k=request.top_k)

    return {
        "type": "rag_response",
        "query": response.query,
        "answer": response.answer,
        "sources": response.sources,
        "detected_intent": analysis.detected_intent,
        "key_terms": analysis.key_terms,
    }
