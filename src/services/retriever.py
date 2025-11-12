import psycopg2
from pgvector.psycopg2 import register_vector


class Retriever:

    def __init__(self, dbname, user, password):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password)

        register_vector(self.conn)
        self.cur = self.conn.cursor()
    
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
        self.cur.execute(search_sql, (query_embedding, query_embedding, limit))

        results = self.cur.fetchall()

        return results


