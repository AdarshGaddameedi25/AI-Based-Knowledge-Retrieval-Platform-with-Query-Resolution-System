# Milestone 4 Analytics & Telemetry Reference

## Overview
The Milestone 4 Analytics Engine provides real-time query performance monitoring, retrieval quality metrics, and automated knowledge gap detection.

## REST API Specification

### `GET /api/analytics/overview`
Returns high-level KPI summary metrics over the specified timeframe.
- **Parameters**: `days` (int, default: 30)
- **Response Keys**: `total_queries`, `answered_queries`, `knowledge_gap_queries`, `clarification_queries`, `retrieval_accuracy_rate`, `avg_similarity_score`, `avg_retrieved_chunks`, `avg_latency_ms`.

### `GET /api/analytics/queries`
Returns a paginated log of recorded query analytics transactions.
- **Parameters**: `page` (int), `limit` (int), `domain` (str), `status` (str), `search` (str)
- **Response Keys**: `records` (list of query analytics objects), `total`, `page`, `pages`.

### `GET /api/analytics/gaps`
Returns tracked missing knowledge topics sorted by frequency.
- **Parameters**: `limit` (int, default: 20), `domain` (str)
- **Response Keys**: `gaps` (list of knowledge gap objects).

### `GET /api/analytics/domains`
Returns distribution of queries across HR, Technology, Finance, and other domains.

### `GET /api/analytics/types`
Returns distribution of query intents (factual, procedural, comparative, ambiguous, direct).

### `GET /api/analytics/terms`
Returns most frequently occurring key terms extracted from user queries.
