"""End-to-end RAG: retrieve -> prompt Claude -> return grounded answer.

The system prompt explicitly forbids using outside knowledge — this is what
turns "Claude with extra text" into a real RAG system. If the retrieved
context doesn't answer the question, Claude says so.
"""
from dataclasses import dataclass
from typing import List

import anthropic

from src.config import ANTHROPIC_API_KEY, CHAT_MODEL
from src.retrieve import Hit, search

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a legal assistant answering questions about Indian Acts.

Rules:
- Use ONLY the numbered context passages provided.
- If the context does not contain the answer, say so explicitly. Do not guess.
- Cite passages by their number, e.g. "[1]", "[2]".
- Keep answers concise and quote the relevant statutory text when helpful.
"""


@dataclass
class RagAnswer:
    answer: str
    hits: List[Hit]


def _format_context(hits: List[Hit]) -> str:
    """Pack the retrieved chunks into a numbered block the LLM can cite."""
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[{i}] {h.title}, chunk #{h.chunk_index}\n{h.chunk_text}"
        )
    return "\n\n".join(parts)


def ask(question: str, k: int = 5, max_tokens: int = 600) -> RagAnswer:
    hits = search(question, k=k)
    context = _format_context(hits)

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = _client.messages.create(
        model=CHAT_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return RagAnswer(answer=answer_text, hits=hits)
