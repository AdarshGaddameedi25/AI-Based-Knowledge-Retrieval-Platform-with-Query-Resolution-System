"""
backend/api/routes/analytics.py — M4.1 Analytics API

Endpoints:
  GET /api/analytics/overview       — summary metrics
  GET /api/analytics/queries        — paginated query list with filters
  GET /api/analytics/domains        — domain distribution
  GET /api/analytics/query-types    — query-type distribution
  GET /api/analytics/knowledge-gaps — paginated knowledge gap list
  GET /api/analytics/trends         — daily trends
  GET /api/analytics/terms          — frequently queried terms

All endpoints:
  - Use real recorded data only — no synthetic values
  - Support date range via ?days=N parameter
  - Support domain filter via ?domain=...
  - Validate all query parameters
  - Return structured JSON with error handling
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.services.analytics_service import analytics_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/analytics/overview")
def get_analytics_overview(
    days: int = QueryParam(default=30, ge=1, le=365, description="Number of days to aggregate"),
    db: Session = Depends(get_db),
):
    """
    High-level analytics summary.

    Returns:
      total_queries, answered_queries, knowledge_gap_queries,
      clarification_queries, low_confidence_queries,
      avg_retrieved_chunks, avg_similarity_score,
      avg_response_latency_ms, retrieval_success_rate
    """
    try:
        return analytics_service.get_overview(db, days=days)
    except Exception as e:
        logger.error("Analytics overview error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics overview.")


@router.get("/analytics/queries")
def get_analytics_queries(
    limit: int = QueryParam(default=20, ge=1, le=100),
    offset: int = QueryParam(default=0, ge=0),
    domain: str = QueryParam(default=None, description="Filter by domain (hr, technology, legal)"),
    query_type: str = QueryParam(default=None, description="Filter by query type (factual, procedural, comparative, ambiguous)"),
    knowledge_gap_only: bool = QueryParam(default=False, description="Show only knowledge-gap queries"),
    days: int = QueryParam(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Paginated list of recorded queries with analytics metadata.

    Supports filtering by domain, query_type, knowledge_gap status.
    """
    try:
        return analytics_service.get_recent_queries(
            db,
            limit=limit,
            offset=offset,
            domain=domain,
            query_type=query_type,
            knowledge_gap_only=knowledge_gap_only,
            days=days,
        )
    except Exception as e:
        logger.error("Analytics queries error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics queries.")


@router.get("/analytics/domains")
def get_analytics_domains(
    days: int = QueryParam(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Query count grouped by detected domain.
    """
    try:
        return {"domains": analytics_service.get_domain_distribution(db, days=days)}
    except Exception as e:
        logger.error("Analytics domains error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve domain distribution.")


@router.get("/analytics/query-types")
def get_analytics_query_types(
    days: int = QueryParam(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Query count grouped by query_type (factual, procedural, comparative, ambiguous, direct).
    """
    try:
        return {"query_types": analytics_service.get_query_type_distribution(db, days=days)}
    except Exception as e:
        logger.error("Analytics query-types error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve query type distribution.")


@router.get("/analytics/knowledge-gaps")
def get_analytics_knowledge_gaps(
    limit: int = QueryParam(default=20, ge=1, le=100),
    offset: int = QueryParam(default=0, ge=0),
    domain: str = QueryParam(default=None, description="Filter by domain"),
    min_frequency: int = QueryParam(default=1, ge=1, description="Minimum occurrence frequency"),
    db: Session = Depends(get_db),
):
    """
    Paginated knowledge gap records.

    Shows queries that repeatedly fail to find sufficient evidence.
    Ordered by frequency (most frequent first).
    """
    try:
        return analytics_service.get_knowledge_gaps(
            db,
            limit=limit,
            offset=offset,
            domain=domain,
            min_frequency=min_frequency,
        )
    except Exception as e:
        logger.error("Analytics knowledge-gaps error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge gaps.")


@router.get("/analytics/trends")
def get_analytics_trends(
    days: int = QueryParam(default=14, ge=1, le=90),
    db: Session = Depends(get_db),
):
    """
    Daily query volume, knowledge gap count, and average similarity score.
    Useful for trend visualization.
    """
    try:
        return {"trends": analytics_service.get_daily_trends(db, days=days)}
    except Exception as e:
        logger.error("Analytics trends error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve trends.")


@router.get("/analytics/terms")
def get_analytics_terms(
    days: int = QueryParam(default=30, ge=1, le=365),
    top_n: int = QueryParam(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Most frequently occurring key terms across all queries.
    """
    try:
        return {"terms": analytics_service.get_common_terms(db, days=days, top_n=top_n)}
    except Exception as e:
        logger.error("Analytics terms error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve common terms.")
