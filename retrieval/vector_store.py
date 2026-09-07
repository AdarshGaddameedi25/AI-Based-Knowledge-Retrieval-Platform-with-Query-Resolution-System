import logging
from typing import Optional, List
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from backend.db.models import Embedding, DocumentChunk, Document

logger = logging.getLogger(__name__)


class VectorStore:
    def similarity_search(
        self,
        db: Session,
        query_embedding: List[float],
        top_k: int = 5,
        domain_filter: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> List[dict]:
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Build domain filter clause
        domain_clause = ""
        params: dict = {"embedding": embedding_str, "top_k": top_k}

        if domain_filter:
            domain_clause = "AND LOWER(d.domain) = LOWER(:domain_filter)"
            params["domain_filter"] = domain_filter

        sql = text(f"""
            SELECT
                e.chunk_id,
                e.embedding_model,
                dc.document_id,
                dc.text,
                dc.chunk_index,
                dc.page_number,
                dc.chunk_metadata,
                d.file_name,
                d.domain,
                1 - (e.embedding <=> CAST(:embedding AS vector)) AS similarity_score
            FROM embeddings e
            JOIN document_chunks dc ON e.chunk_id = dc.chunk_id
            JOIN documents d ON dc.document_id = d.document_id
            WHERE d.status = 'indexed'
            {domain_clause}
            ORDER BY e.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        rows = db.execute(sql, params).fetchall()

        results = []
        for rank, row in enumerate(rows, start=1):
            score = float(row.similarity_score)
            if score < similarity_threshold:
                continue
            results.append({
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "source_file": row.file_name,
                "text": row.text,
                "similarity_score": score,
                "rank": rank,
                "chunk_index": row.chunk_index,
                "page_number": row.page_number,
                "domain": row.domain,
                "metadata": dict(row.chunk_metadata) if row.chunk_metadata else {},
            })

        return results

    def count_chunks(self, db: Session) -> int:
        return db.query(func.count(Embedding.embedding_id)).scalar()

    def count_documents(self, db: Session) -> int:
        return db.query(func.count(Document.document_id)).filter(Document.status == "indexed").scalar()

    def get_domain_stats(self, db: Session) -> List[dict]:
        """Return document and chunk counts grouped by domain."""
        rows = (
            db.query(
                Document.domain,
                func.count(Document.document_id).label("document_count"),
            )
            .filter(Document.status == "indexed")
            .group_by(Document.domain)
            .all()
        )
        return [
            {"domain": row.domain or "unspecified", "document_count": row.document_count}
            for row in rows
        ]
