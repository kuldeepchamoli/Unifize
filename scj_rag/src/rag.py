import anthropic
from .config import ANTHROPIC_API_KEY
from .retrieve import search

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM = """You are an assistant answering questions about Indian Supreme Court
judgments from 2025. Use ONLY the provided excerpts. After every factual claim,
cite the source as [case_id, decision_date]. If the excerpts do not contain the
answer, say so plainly — do not guess."""

def answer(question: str, k: int = 5) -> str:
    hits = search(question, k=k)
    if not hits:
        return "No relevant judgments found."

    context = "\n\n---\n\n".join(
        f"[{h['case_id']}, {h['date']}] {h['title']}\n{h['text']}"
        for h in hits
    )
    prompt = f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer:"

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
