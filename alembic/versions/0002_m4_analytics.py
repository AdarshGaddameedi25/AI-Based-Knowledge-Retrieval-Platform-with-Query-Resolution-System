"""M4 Analytics tables: query_analytics and knowledge_gaps

Revision ID: 0002_m4_analytics
Revises: 0001_initial
Create Date: 2026-10-03

Non-destructive migration — only adds new tables.
Does NOT modify existing M1/M2/M3 tables.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002_m4_analytics"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── query_analytics ────────────────────────────────────────────────────
    op.create_table(
        "query_analytics",
        sa.Column("analytics_id", sa.String(), nullable=False),
        sa.Column("query_id", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),

        # Query text
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=True),

        # Classification
        sa.Column("query_type", sa.String(), nullable=True),
        sa.Column("detected_intent", sa.String(), nullable=True),
        sa.Column("detected_domain", sa.String(), nullable=True),

        # Routing
        sa.Column("routing_path", sa.String(), nullable=True),

        # Clarification
        sa.Column("clarification_required", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("clarification_count", sa.Integer(), nullable=False, server_default="0"),

        # Retrieval
        sa.Column("retrieval_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieved_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_similarity_score", sa.Float(), nullable=True),
        sa.Column("avg_similarity_score", sa.Float(), nullable=True),
        sa.Column("source_count", sa.Integer(), nullable=False, server_default="0"),

        # Response
        sa.Column("response_status", sa.String(), nullable=True),
        sa.Column("response_generated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("knowledge_gap", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("knowledge_gap_reason", sa.Text(), nullable=True),

        # Performance
        sa.Column("response_latency_ms", sa.Float(), nullable=True),

        # Key terms (JSON list)
        sa.Column("key_terms", sa.JSON(), nullable=True),

        sa.PrimaryKeyConstraint("analytics_id"),
    )
    op.create_index("ix_qa_query_id", "query_analytics", ["query_id"])
    op.create_index("ix_qa_session_id", "query_analytics", ["session_id"])
    op.create_index("ix_qa_timestamp", "query_analytics", ["timestamp"])
    op.create_index("ix_qa_detected_domain", "query_analytics", ["detected_domain"])
    op.create_index("ix_qa_knowledge_gap", "query_analytics", ["knowledge_gap"])

    # ── knowledge_gaps ─────────────────────────────────────────────────────
    op.create_table(
        "knowledge_gaps",
        sa.Column("gap_id", sa.String(), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("first_seen", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("frequency", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("best_similarity_score", sa.Float(), nullable=True),
        sa.Column("retrieval_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="detected"),
        sa.PrimaryKeyConstraint("gap_id"),
    )
    op.create_index("ix_kg_normalized_query", "knowledge_gaps", ["normalized_query"])
    op.create_index("ix_kg_domain", "knowledge_gaps", ["domain"])


def downgrade() -> None:
    op.drop_index("ix_kg_domain", table_name="knowledge_gaps")
    op.drop_index("ix_kg_normalized_query", table_name="knowledge_gaps")
    op.drop_table("knowledge_gaps")

    op.drop_index("ix_qa_knowledge_gap", table_name="query_analytics")
    op.drop_index("ix_qa_detected_domain", table_name="query_analytics")
    op.drop_index("ix_qa_timestamp", table_name="query_analytics")
    op.drop_index("ix_qa_session_id", table_name="query_analytics")
    op.drop_index("ix_qa_query_id", table_name="query_analytics")
    op.drop_table("query_analytics")
