from dataclasses import dataclass, field
import openai

from config.settings import settings
from retrieval.retriever import Retriever, RetrievalResult


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[dict]
    context_used: str
    retrieval_results: list[RetrievalResult]


class RAGPipeline:
    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say: "
        "'I could not find an answer to your question in the available knowledge base.' "
        "Do not invent information. Cite the source document when possible."
    )

    def __init__(self, retriever: Retriever = None):
        if not settings.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._retriever = retriever or Retriever()

    def _build_context(self, results: list[RetrievalResult]) -> str:
        sections = []
        for r in results:
            sections.append(
                f"[Source: {r.source_file} | Chunk {r.chunk_index} | Score: {r.similarity_score:.3f}]\n{r.text}"
            )
        return "\n\n---\n\n".join(sections)

    def run(self, query: str, top_k: int = None) -> RAGResponse:
        results = self._retriever.retrieve(query, top_k=top_k)

        if not results:
            return RAGResponse(
                query=query,
                answer="I could not find an answer to your question in the available knowledge base.",
                sources=[],
                context_used="",
                retrieval_results=[],
            )

        context = self._build_context(results)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        completion = self._client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        answer = completion.choices[0].message.content.strip()

        sources = [
            {
                "source_file": r.source_file,
                "chunk_id": r.chunk_id,
                "chunk_index": r.chunk_index,
                "similarity_score": r.similarity_score,
                "rank": r.rank,
            }
            for r in results
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            context_used=context,
            retrieval_results=results,
        )
