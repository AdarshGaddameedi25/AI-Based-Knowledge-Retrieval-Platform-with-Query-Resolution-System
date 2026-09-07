import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.db.base import get_db_context
from backend.services.ingestion_service import IngestionService
from retrieval.rag_pipeline import RAGPipeline
from agents.query_understanding_agent import QueryUnderstandingAgent

def run_e2e():
    print("=== STEP 1: Ingesting Sample Documents ===")
    service = IngestionService()
    docs_to_ingest = [
        "data/sample_documents/hr/employee_handbook.txt",
        "data/sample_documents/hr/leave_policy.txt",
        "data/sample_documents/technology/cloud_security.txt",
    ]
    
    with get_db_context() as db:
        for doc_path in docs_to_ingest:
            abs_path = os.path.abspath(doc_path)
            if os.path.exists(abs_path):
                print(f"Ingesting: {doc_path}")
                res = service.ingest_file(abs_path, db=db, original_filename=os.path.basename(doc_path))
                print(f"  -> Ingested doc_id={res['document_id']}, chunks={res['chunks_created']}, status={res['status']}")
            else:
                print(f"File not found: {abs_path}")

    print("\n=== STEP 2: Querying via RAG Pipeline (PostgreSQL pgvector + OpenRouter) ===")
    query_agent = QueryUnderstandingAgent()
    pipeline = RAGPipeline()
    
    test_queries = [
        "What is the policy for parental leave and how many weeks are allowed?",
        "What are the encryption standards required for cloud data at rest?",
        "Hello, can you help me?",
    ]
    
    with get_db_context() as db:
        for q in test_queries:
            print(f"\n--- Query: '{q}' ---")
            analysis = query_agent.analyze(q)
            print(f"Intent: {analysis.detected_intent} | Requires Retrieval: {analysis.requires_retrieval}")
            
            if not analysis.requires_retrieval:
                print("Direct response: Hello! I am the AI Knowledge Retrieval Assistant. Ask me anything about the uploaded documents.")
                continue
                
            response = pipeline.run(q, db=db, top_k=3)
            print(f"Retrieved Chunks: {len(response.retrieval_results)}")
            for src in response.sources:
                print(f"  - [{src['rank']}] {src['source_file']} (score: {src['similarity_score']})")
            print(f"LLM Answer:\n{response.answer}\n")

if __name__ == '__main__':
    run_e2e()
