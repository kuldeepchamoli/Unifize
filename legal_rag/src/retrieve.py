"""Top-k semantic retrieval against the `chunks` table.

The SQL operator `<=>` is pgvector's cosine distance.
ORDER BY embedding <=> query  ascending  =>  most similar first.
The HNSW index defined in sql/01_schema.sql is what makes this fast.
"""
from dataclasses import dataclass
from typing import List

from src.db import get_conn
from src.embed import embed


@dataclass
class Hit:
    document_id: int
    title: str
    chunk_index: int
    chunk_text: str
    distance: float

    def citation(self) -> str:
        """Short label for citing this chunk in an answer."""
        return f"[{self.title}, chunk #{self.chunk_index}]"


def search(query: str, k: int = 5) -> List[Hit]:
    """Embed the query, run a top-k cosine search, return Hit objects."""
    qvec = embed([query])[0]
    sql = """
        SELECT c.document_id,
               d.title,
               c.chunk_index,
               c.chunk_text,
               c.embedding <=> %s AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (qvec, qvec, k))
        rows = cur.fetchall()

    return [
        Hit(
            document_id=r[0],
            title=r[1],
            chunk_index=r[2],
            chunk_text=r[3],
            distance=float(r[4]),
        )
        for r in rows
    ]
