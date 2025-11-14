from config.database import db

class VectorStore:
    def __init__(self):
        self.db = db
    
    def store_embedding(self, text, embedding, metadata=None):
        """
        Сохраняет текст и его эмбеддинг в БД
        
        Args:
            text: исходный текст
            embedding: векторное представление
            metadata: дополнительные метаданные (источник, тип и т.д.)
        """
        try:
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            with self.db.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO text_embeddings 
                    (text_content, embedding, metadata) 
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (text, embedding_list, metadata or {}))
                
                inserted_id = cursor.fetchone()['id']
                self.db.connection.commit()
                
            print(f"✅ Сохранен эмбеддинг с ID: {inserted_id}")
            return inserted_id
            
        except Exception as e:
            print(f"❌ Ошибка сохранения эмбеддинга: {e}")
            self.db.connection.rollback()
            return None
    
    def get_all_texts(self, limit=100):
        """Получить все тексты (для отладки)"""
        try:
            with self.db.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, text_content, metadata, created_at
                    FROM text_embeddings
                    ORDER BY created_at DESC
                    LIMIT %s
                """, [limit])
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"❌ Ошибка получения текстов: {e}")
            return []