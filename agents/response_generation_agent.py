from dataclasses import dataclass, field
import openai

from config.settings import settings
from retrieval.retriever import RetrievalResult


@dataclass
class GeneratedResponse:
    query: str
    answer: str
    citations: list[dict]
    context_chunks_used: int


class ResponseGenerationAgent:
    SYSTEM_PROMPT = (
        "You are a knowledgeable AI assistant. Answer using ONLY the context provided. "
        "If the answer is not in the context, respond with: "
        "'The requested information is not available in the knowledge base.' "
        "Always cite the source document name at the end of your answer."
    )

    def __init__(self):
        if not settings.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def generate(self, query: str, retrieval_results: list[RetrievalResult]) -> GeneratedResponse:
        if not retrieval_results:
            return GeneratedResponse(
                query=query,
                answer="The requested information is not available in the knowledge base.",
                citations=[],
                context_chunks_used=0,
            )

        context_parts = []
        for r in retrieval_results:
            context_parts.append(f"[{r.source_file}]\n{r.text}")
        context = "\n\n".join(context_parts)

        completion = self._client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.1,
        )

        answer = completion.choices[0].message.content.strip()
        citations = [
            {"source": r.source_file, "chunk_id": r.chunk_id, "score": r.similarity_score}
            for r in retrieval_results
        ]

        return GeneratedResponse(
            query=query,
            answer=answer,
            citations=citations,
            context_chunks_used=len(retrieval_results),
        )
