import numpy as np
from typing import List, Dict, Any, Optional
from src.services.vector_store import VectorStore


class Embeder:
    def __init__(self, model: Any, store: VectorStore):
        self.model = model
        self.store = store

        # Тестируем модель и получаем правильную размерность
        test_embedding = self._safe_embed("test")
        self.embedding_dim = len(test_embedding)

        print(f"✅ Эмбеддер инициализирован. Размерность: {self.embedding_dim}")

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
            return self.store.store_embedding(content, embedding, metadata)

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
        res = []
        for doc, metadata in zip(documents, metadata_list):
            res.append(self.embed_and_store(doc, metadata))

        return res