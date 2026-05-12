from .db import connect
from .embed import embed

def search(question: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks most relevant to the question."""
    q_vec = embed([question])[0]
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.text, c.case_id, j.title, j.decision_date, j.citation,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN judgments j USING (case_id)
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (q_vec, q_vec, k))
        rows = cur.fetchall()
    conn.close()
    return [
        {"text": r[0], "case_id": r[1], "title": r[2],
         "date": r[3], "citation": r[4], "similarity": r[5]}
        for r in rows
    ]
