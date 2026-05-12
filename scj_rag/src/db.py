import psycopg
from pgvector.psycopg import register_vector
from src.config import DATABASE_URL

def connect():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn