"""
analytics_service.py — M4.1 Query Analytics & Knowledge Gap Detection

Responsibilities:
  1. Record a QueryAnalytics entry for every orchestrator response.
  2. Detect knowledge gaps using evidence-based rules (not just short answers).
  3. Upsert/group repeated knowledge gaps in the knowledge_gaps table.
  4. Provide aggregation helpers for the analytics API.

Knowledge-gap detection rules (evidence-based, per M4 spec):
  A. No chunks retrieved (retrieval_count == 0)
  B. Retrieved evidence below similarity threshold (best_score < threshold)
  C. Repeated query with same insufficient evidence (frequency > 1)
  D. Low-confidence combined with very few chunks
  NOT a gap: short answer alone, or direct/clarification routing
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, Integer as sa_Integer

from backend.db.models import QueryAnalytics, KnowledgeGap

logger = logging.getLogger(__name__)

# Similarity threshold below which evidence is considered insufficient for a gap
_GAP_SIMILARITY_THRESHOLD = 0.25
# Max chunks that, combined with low similarity, still qualify as a gap
_GAP_LOW_CHUNK_THRESHOLD = 1
# Normalize query for deduplication (remove punctuation, lowercase, strip)
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_for_dedup(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — used for gap grouping."""
    cleaned = _PUNCT_RE.sub(" ", query.lower().strip())
    return re.sub(r"\s+", " ", cleaned).strip()


class AnalyticsService:
    """
    Records and aggregates query analytics.
    All writes are best-effort — failures are logged but do not break the query pipeline.
    """

    def record_query(
        self,
        db: Session,
        *,
        original_query: str,
        normalized_query: str = "",
        query_type: str = "",
        detected_intent: str = "",
        detected_domain: Optional[str] = None,
        routing_path: str = "",
        clarification_required: bool = False,
        retrieval_count: int = 0,
        retrieved_chunks: int = 0,
        similarity_scores: Optional[List[float]] = None,
        source_count: int = 0,
        response_status: str = "answered",
        response_generated: bool = True,
        response_latency_ms: Optional[float] = None,
        key_terms: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> Optional[QueryAnalytics]:
        """
        Insert a QueryAnalytics record.
        Also detects and records knowledge gaps as a side effect.
        Returns the created record, or None on failure.
        """
        try:
            scores = similarity_scores or []
            best_score = max(scores) if scores else None
            avg_score = (sum(scores) / len(scores)) if scores else None

            is_gap, gap_reason = self._detect_gap(
                routing_path=routing_path,
                retrieval_count=retrieval_count,
                best_score=best_score,
                response_status=response_status,
            )

            record = QueryAnalytics(
                query_id=query_id,
                session_id=session_id,
                original_query=original_query[:2000],
                normalized_query=(normalized_query or original_query)[:2000],
                query_type=query_type,
                detected_intent=detected_intent,
                detected_domain=detected_domain,
                routing_path=routing_path,
                clarification_required=clarification_required,
                clarification_count=1 if clarification_required else 0,
                retrieval_count=retrieval_count,
                retrieved_chunks=retrieved_chunks,
                best_similarity_score=best_score,
                avg_similarity_score=avg_score,
                source_count=source_count,
                response_status=response_status,
                response_generated=response_generated,
                knowledge_gap=is_gap,
                knowledge_gap_reason=gap_reason,
                response_latency_ms=response_latency_ms,
                key_terms=key_terms or [],
            )
            db.add(record)

            # Record/update knowledge gap entry if detected
            if is_gap:
                self._upsert_knowledge_gap(
                    db=db,
                    original_query=original_query,
                    domain=detected_domain,
                    reason=gap_reason or "insufficient_evidence",
                    retrieval_count=retrieval_count,
                    best_score=best_score,
                    session_id=session_id,
                )

            db.commit()
            logger.debug("[Analytics] Recorded analytics for query: %r", original_query[:60])
            return record

        except Exception as e:
            logger.warning("[Analytics] Failed to record analytics: %s", e)
            try:
                db.rollback()
            except Exception:
                pass
            return None

    def _detect_gap(
        self,
        routing_path: str,
        retrieval_count: int,
        best_score: Optional[float],
        response_status: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Evidence-based gap detection.
        Returns (is_gap: bool, reason: str | None).
        """
        # Not a gap if routing bypassed retrieval
        if routing_path in ("direct", "clarification"):
            return False, None

        # Rule A: No chunks retrieved at all
        if retrieval_count == 0:
            return True, "no_chunks_retrieved"

        # Rule B: Best score below configured threshold
        if best_score is not None and best_score < _GAP_SIMILARITY_THRESHOLD:
            return True, "below_similarity_threshold"

        # Rule D: Response status explicitly marked as gap
        if response_status == "knowledge_gap":
            return True, "response_marked_as_gap"

        # Rule D: Very low chunk count combined with low similarity
        if retrieval_count <= _GAP_LOW_CHUNK_THRESHOLD and best_score is not None and best_score < 0.35:
            return True, "low_evidence_low_confidence"

        return False, None

    def _upsert_knowledge_gap(
        self,
        db: Session,
        original_query: str,
        domain: Optional[str],
        reason: str,
        retrieval_count: int,
        best_score: Optional[float],
        session_id: Optional[str],
    ) -> None:
        """
        Insert or update a KnowledgeGap record.
        Deduplication is based on normalized query + domain.
        """
        norm_q = _normalize_for_dedup(original_query)

        existing = (
            db.query(KnowledgeGap)
            .filter(
                func.lower(KnowledgeGap.normalized_query) == func.lower(norm_q),
            )
            .first()
        )

        if existing:
            # Update existing gap — increment frequency, refresh last_seen
            existing.frequency += 1
            existing.last_seen = datetime.utcnow()
            # Keep the best score (highest similarity seen — partial evidence)
            if best_score is not None:
                if existing.best_similarity_score is None or best_score > existing.best_similarity_score:
                    existing.best_similarity_score = best_score
            curr_cnt = existing.retrieval_count if isinstance(existing.retrieval_count, int) else 0
            existing.retrieval_count = max(curr_cnt, retrieval_count)
            logger.debug(
                "[Analytics] Updated knowledge gap freq=%d for: %r",
                existing.frequency, norm_q[:60],
            )
        else:
            gap = KnowledgeGap(
                normalized_query=norm_q[:2000],
                original_query=original_query[:2000],
                domain=domain,
                reason=reason,
                best_similarity_score=best_score,
                retrieval_count=retrieval_count,
                session_id=session_id,
                status="detected",
            )
            db.add(gap)
            logger.debug("[Analytics] New knowledge gap recorded: %r", norm_q[:60])

    # ──────────────────────────────────────────────────────────────────────
    # Aggregation helpers (for the analytics API)
    # ──────────────────────────────────────────────────────────────────────

    def get_overview(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """
        High-level summary metrics.
        Only uses data from the last `days` days.
        """
        since = datetime.utcnow() - timedelta(days=days)
        base = db.query(QueryAnalytics).filter(QueryAnalytics.timestamp >= since)
        total = base.count()

        answered = base.filter(QueryAnalytics.response_status == "answered").count()
        gaps = base.filter(QueryAnalytics.knowledge_gap == True).count()
        clarifications = base.filter(QueryAnalytics.clarification_required == True).count()
        low_confidence = base.filter(
            QueryAnalytics.best_similarity_score.isnot(None),
            QueryAnalytics.best_similarity_score < 0.4,
        ).count()

        avg_chunks = (
            db.query(func.avg(QueryAnalytics.retrieved_chunks))
            .filter(QueryAnalytics.timestamp >= since)
            .scalar()
        )
        avg_score = (
            db.query(func.avg(QueryAnalytics.avg_similarity_score))
            .filter(
                QueryAnalytics.timestamp >= since,
                QueryAnalytics.avg_similarity_score.isnot(None),
            )
            .scalar()
        )
        avg_latency = (
            db.query(func.avg(QueryAnalytics.response_latency_ms))
            .filter(
                QueryAnalytics.timestamp >= since,
                QueryAnalytics.response_latency_ms.isnot(None),
            )
            .scalar()
        )

        retrieval_queries = base.filter(QueryAnalytics.routing_path == "retrieval").count()
        retrieval_success = base.filter(
            QueryAnalytics.routing_path == "retrieval",
            QueryAnalytics.knowledge_gap == False,
            QueryAnalytics.retrieved_chunks > 0,
        ).count()

        if isinstance(retrieval_queries, (int, float)) and retrieval_queries > 0:
            retrieval_rate = (retrieval_success / retrieval_queries * 100)
        else:
            retrieval_rate = 0.0

        return {
            "period_days": days,
            "total_queries": total,
            "answered_queries": answered,
            "knowledge_gap_queries": gaps,
            "clarification_queries": clarifications,
            "low_confidence_queries": low_confidence,
            "avg_retrieved_chunks": round(avg_chunks or 0, 2),
            "avg_similarity_score": round(avg_score or 0, 4),
            "avg_response_latency_ms": round(avg_latency or 0, 1),
            "retrieval_success_rate": round(retrieval_rate, 1),
        }

    def get_query_type_distribution(self, db: Session, days: int = 30) -> List[Dict]:
        since = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(QueryAnalytics.query_type, func.count(QueryAnalytics.analytics_id).label("count"))
            .filter(QueryAnalytics.timestamp >= since, QueryAnalytics.query_type.isnot(None))
            .group_by(QueryAnalytics.query_type)
            .order_by(desc("count"))
            .all()
        )
        return [{"query_type": r.query_type, "count": r.count} for r in rows]

    def get_domain_distribution(self, db: Session, days: int = 30) -> List[Dict]:
        since = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(QueryAnalytics.detected_domain, func.count(QueryAnalytics.analytics_id).label("count"))
            .filter(QueryAnalytics.timestamp >= since)
            .group_by(QueryAnalytics.detected_domain)
            .order_by(desc("count"))
            .all()
        )
        return [
            {"domain": r.detected_domain or "unspecified", "count": r.count}
            for r in rows
        ]

    def get_recent_queries(
        self,
        db: Session,
        limit: int = 20,
        offset: int = 0,
        domain: Optional[str] = None,
        query_type: Optional[str] = None,
        knowledge_gap_only: bool = False,
        days: int = 30,
    ) -> Dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=days)
        q = db.query(QueryAnalytics).filter(QueryAnalytics.timestamp >= since)
        if domain:
            q = q.filter(func.lower(QueryAnalytics.detected_domain) == domain.lower())
        if query_type:
            q = q.filter(QueryAnalytics.query_type == query_type)
        if knowledge_gap_only:
            q = q.filter(QueryAnalytics.knowledge_gap == True)

        total = q.count()
        rows = q.order_by(desc(QueryAnalytics.timestamp)).offset(offset).limit(limit).all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "queries": [self._serialize_analytics(r) for r in rows],
        }

    def get_daily_trends(self, db: Session, days: int = 14) -> List[Dict]:
        since = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(
                func.date(QueryAnalytics.timestamp).label("date"),
                func.count(QueryAnalytics.analytics_id).label("total"),
                func.sum(QueryAnalytics.knowledge_gap.cast(sa_Integer())).label("gaps"),
                func.avg(QueryAnalytics.avg_similarity_score).label("avg_score"),
            )
            .filter(QueryAnalytics.timestamp >= since)
            .group_by(func.date(QueryAnalytics.timestamp))
            .order_by(func.date(QueryAnalytics.timestamp))
            .all()
        )
        return [
            {
                "date": str(r.date),
                "total_queries": r.total,
                "knowledge_gaps": r.gaps or 0,
                "avg_similarity": round(float(r.avg_score or 0), 4),
            }
            for r in rows
        ]

    def get_knowledge_gaps(
        self,
        db: Session,
        limit: int = 20,
        offset: int = 0,
        domain: Optional[str] = None,
        min_frequency: int = 1,
    ) -> Dict[str, Any]:
        q = db.query(KnowledgeGap)
        if domain:
            q = q.filter(func.lower(KnowledgeGap.domain) == domain.lower())
        if min_frequency > 1:
            q = q.filter(KnowledgeGap.frequency >= min_frequency)

        total = q.count()
        rows = q.order_by(desc(KnowledgeGap.frequency), desc(KnowledgeGap.last_seen)).offset(offset).limit(limit).all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "gaps": [self._serialize_gap(r) for r in rows],
        }

    def get_common_terms(self, db: Session, days: int = 30, top_n: int = 20) -> List[Dict]:
        """Aggregate key_terms from analytics records to find frequently queried topics."""
        since = datetime.utcnow() - timedelta(days=days)
        rows = (
            db.query(QueryAnalytics.key_terms)
            .filter(
                QueryAnalytics.timestamp >= since,
                QueryAnalytics.key_terms.isnot(None),
            )
            .all()
        )
        term_counts: Dict[str, int] = {}
        for (terms,) in rows:
            if isinstance(terms, list):
                for t in terms:
                    if t and len(t) > 2:
                        term_counts[t] = term_counts.get(t, 0) + 1
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"term": t, "count": c} for t, c in sorted_terms]

    # ── serializers ──────────────────────────────────────────────────────

    @staticmethod
    def _serialize_analytics(r: QueryAnalytics) -> Dict:
        return {
            "analytics_id": r.analytics_id,
            "query_id": r.query_id,
            "session_id": r.session_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "original_query": r.original_query,
            "normalized_query": r.normalized_query,
            "query_type": r.query_type,
            "detected_intent": r.detected_intent,
            "detected_domain": r.detected_domain,
            "routing_path": r.routing_path,
            "clarification_required": r.clarification_required,
            "retrieval_count": r.retrieval_count,
            "retrieved_chunks": r.retrieved_chunks,
            "best_similarity_score": r.best_similarity_score,
            "avg_similarity_score": r.avg_similarity_score,
            "source_count": r.source_count,
            "response_status": r.response_status,
            "response_generated": r.response_generated,
            "knowledge_gap": r.knowledge_gap,
            "knowledge_gap_reason": r.knowledge_gap_reason,
            "response_latency_ms": r.response_latency_ms,
            "key_terms": r.key_terms or [],
        }

    @staticmethod
    def _serialize_gap(r: KnowledgeGap) -> Dict:
        return {
            "gap_id": r.gap_id,
            "original_query": r.original_query,
            "normalized_query": r.normalized_query,
            "domain": r.domain,
            "reason": r.reason,
            "first_seen": r.first_seen.isoformat() if r.first_seen else None,
            "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            "frequency": r.frequency,
            "best_similarity_score": r.best_similarity_score,
            "retrieval_count": r.retrieval_count,
            "status": r.status,
        }


# Application-level singleton
analytics_service = AnalyticsService()
