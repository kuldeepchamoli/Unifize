import anthropic
from pathlib import Path
from .config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS
from .retrieve import search

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "f3_poc_v0.0.md"
SYSTEM = _PROMPT_FILE.read_text().strip()


def answer(question: str, k: int = 5) -> str:
    hits = search(question, k=k)
    if not hits:
        return "No relevant judgments found."

    context = "\n\n---\n\n".join(
        f"[{h['case_id']}, {h['date']}] {h['title']}\n{h['text']}"
        for h in hits
    )
    prompt = f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer:"
    msg = _create_message(prompt)
    return msg.content[0].text


def _create_message(prompt: str):
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
