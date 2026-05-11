"""Postgres connection helper.

Why this exists:
- Centralises the connection string (so we change PGHOST in .env once and
  every script picks it up).
- Registers the `vector` Python <-> SQL adapter, so a numpy array or list
  of floats can be passed directly into queries against VECTOR columns.
"""
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import register_vector

from src.config import PG_CONN_STR


@contextmanager
def get_conn():
    """Yield a connection with the pgvector type registered.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    conn = psycopg.connect(PG_CONN_STR)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
