import numpy as np
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values


class Embeder:
    """
    Класс эмбеддер для преобразования входящих запросов.
    """

    def __init__(self, model: Any, db_config: Dict[str, Any]):
        """
        Инициализация эмбеддера.
        Args:
            model (Any): Модель для создания эмбеддингов
            db_config (Dict): Конфигурация подключения к PostgreSQL
        """

        self.model = model

        test_embedding = self._safe_embed("test")
        self.embedding_dim = len(test_embedding)

        self.db_config = db_config

        self._init_database()

        print(f"✅ Эмбеддер инициализирован. Размерность: {self.embedding_dim}")

    def _get_connection(self):
        """Создает соединение с БД"""
        return psycopg2.connect(**self.db_config)

    def _init_database(self):
        """Инициализирует БД"""
        try:
            conn = self._get_connection()
            conn.close()
            print("✅ База данных инициализирована")

        except Exception as e:
            print(f"❌ Ошибка инициализации БД: {e}")
            raise

    def _safe_embed(self, text: str) -> np.ndarray:
        """Безопасное извлечение эмбеддинга"""
        embedding = self.model.embed(text)

        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], (list, np.ndarray)):
                embedding_array = np.array(embedding[0], dtype=np.float32)
            else:
                embedding_array = np.array(embedding, dtype=np.float32)
        else:
            embedding_array = np.array(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        return embedding_array

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Генерация эмбеддинга для текста.
        Args:
            text (str): Входной текст
        Returns:
            np.ndarray: Вектор эмбеддинга
        """
        if not text or not text.strip():
            raise ValueError("Текст не может быть пустым")
        return self._safe_embed(text)

    def embed_and_store(self, content: str, metadata: Optional[Dict] = None) -> int:
        """
        Преобразует текст в эмбеддинг и сохраняет в базу данных.
        Args:
            content (str): Текст для преобразования
            metadata (Dict): Метаданные документа
        Returns:
            int: ID сохраненной записи
        """
        embedding = self.generate_embedding(content)

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_embeddings (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                """, (content, embedding.tolist(), metadata or {}))

                record_id = cur.fetchone()[0]

            conn.commit()
            conn.close()
            print(f"✅ Эмбеддинг сохранен в БД с ID: {record_id}")
            return record_id

        except Exception as e:
            print(f"❌ Ошибка сохранения эмбеддинга: {e}")
            raise

    def batch_embed_and_store(self, documents: List[str], metadata_list: Optional[List[Dict]] = None) -> List[int]:
        """
        Пакетное преобразование текстов в эмбеддинги и сохранение в базу данных.
        Args:
            documents (List[str]): Список текстов для обработки
            metadata_list (List[Dict]): Список метаданных
        Returns:
            List[int]: Список ID сохраненных записей
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)

        if len(documents) != len(metadata_list):
            raise ValueError("Количество документов и метаданных должно совпадать")

        print(f"🔄 Генерация эмбеддингов для {len(documents)} документов...")
        embeddings = [self.generate_embedding(doc) for doc in documents]

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                data = [(doc, emb.tolist(), meta) for doc, emb, meta in zip(documents, embeddings, metadata_list)]

                execute_values(cur, """
                    INSERT INTO document_embeddings (content, embedding, metadata)
                    VALUES %s
                    RETURNING id;
                """, data)

                record_ids = [row[0] for row in cur.fetchall()]

            conn.commit()
            conn.close()
            print(f"✅ Сохранено {len(record_ids)} эмбеддингов в БД")
            return record_ids

        except Exception as e:
            print(f"❌ Ошибка пакетного сохранения эмбеддингов: {e}")
            raise

    def get_embedding_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает запись с эмбеддингом по ID.
        Args:
            record_id (int): ID записи
        Returns:
            Dict: Информация о записи (id, content, embedding, metadata, created_at)
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, embedding, metadata, created_at
                    FROM document_embeddings WHERE id = %s;
                """, (record_id,))

                row = cur.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'content': row[1],
                        'embedding': row[2],
                        'metadata': row[3],
                        'created_at': row[4]
                    }
            conn.close()
            return None

        except Exception as e:
            print(f"❌ Ошибка получения эмбеддинга: {e}")
            return None

    def get_stored_count(self) -> int:
        """Возвращает количество сохраненных эмбеддингов в БД"""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_embeddings;")
                count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"❌ Ошибка получения количества записей: {e}")
            return 0

    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели"""
        return {
            'embedding_dim': self.embedding_dim,
            'stored_embeddings_count': self.get_stored_count()
        }
    