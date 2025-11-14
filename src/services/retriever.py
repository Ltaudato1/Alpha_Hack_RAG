import psycopg2
from pgvector.psycopg2 import register_vector
from config.database import db


class Retriever:

    def __init__(self):
        self.db = db
    
    def retrieve(self, query_embedding, limit=5):

        search_sql = """
        SELECT 
            id,
            title,
            content,
            1 - (embedding <=> %s) as similarity
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
        """

        results = ''

        with self.db.connection.cursor() as cur:
            cur.execute(search_sql, (query_embedding, query_embedding, limit))
            results = cur.fetchall()

        return results


