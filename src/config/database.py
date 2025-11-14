import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Подключение к PostgreSQL с pgvector"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'rag_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', ''),
                cursor_factory=RealDictCursor
            )
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.connection.commit()
            print("✅ Подключение к БД установлено")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise
    
    def disconnect(self):
        """Закрытие соединения с БД"""
        if self.connection:
            self.connection.close()
            print("✅ Подключение к БД закрыто")
    
    def init_tables(self):
        """Инициализация таблиц для хранения векторов"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding vector(768),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx 
                ON text_embeddings 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            self.connection.commit()
        print("✅ Таблицы БД инициализированы")

# Глобальный экземпляр БД
db = Database()