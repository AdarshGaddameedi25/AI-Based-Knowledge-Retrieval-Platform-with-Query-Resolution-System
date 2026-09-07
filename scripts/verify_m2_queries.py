import json
import urllib.request
import sys

queries = [
    {
        "category": "Factual Query",
        "query": "What is the standard maternity leave entitlement under the company policy?",
    },
    {
        "category": "Procedural Query",
        "query": "How to apply for leave and what steps are required?",
    },
    {
        "category": "Comparative Query",
        "query": "What is the difference between maternity leave and paternity leave?",
    },
    {
        "category": "Ambiguous Query",
        "query": "Tell me more about it",
    },
    {
        "category": "Unavailable Information Query",
        "query": "What is the extraterrestrial space travel allowance and lunar base per diem for Mars missions?",
    }
]

url = "http://localhost:8000/api/query"

for item in queries:
    print("=" * 70)
    print(f"CATEGORY: {item['category']}")
    print(f"QUERY:    {item['query']}")
    print("-" * 70)
    
    payload = json.dumps({"query": item["query"], "top_k": 3}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Status:                 SUCCESS (HTTP {resp.status})")
            print(f"Type:                   {data.get('type')}")
            print(f"Detected Intent:        {data.get('detected_intent')}")
            print(f"Detected Domain:        {data.get('detected_domain')}")
            print(f"Clarification Needed:   {data.get('clarification_needed')}")
            if data.get('clarification_needed'):
                print(f"Clarification Question: {data.get('clarification_question')}")
            print(f"Confidence:             {data.get('confidence')}")
            print(f"Retrieval Count:        {data.get('retrieval_count')}")
            print(f"Sources Found:          {len(data.get('sources', []))}")
            for s in data.get('sources', [])[:2]:
                print(f"  - [{s.get('source_file')}] Score: {s.get('similarity_score')} (Page: {s.get('page_number')})")
            print(f"Answer:\n{data.get('answer')[:300]}...")
    except Exception as e:
        print(f"ERROR: {e}")
    print()
