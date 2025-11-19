import psycopg2
from pgvector.psycopg2 import register_vector
from src.config.database import db
from typing import List


class Retriever:
    def __init__(self):
        self.db = db
        # Регистрируем векторный тип
        register_vector(self.db.connection)

    def retrieve(self, query_embedding: List[float], limit: int = 5) -> List:
        search_sql = """
        SELECT DISTINCT metadata, embedding <=> %s AS similarity
        FROM text_embeddings 
        ORDER BY similarity  
        LIMIT %s
        """

        results = ''

        with self.db.connection.cursor() as cur:
            cur.execute(search_sql, (query_embedding, limit))
            results = cur.fetchall()

        return results
