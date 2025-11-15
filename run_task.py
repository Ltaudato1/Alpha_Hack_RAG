# import csv
# import sys
# import os
# import pandas as pd
#
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# sys.path.insert(0, '/app/src')
#
# from config.database import db
# from services.model_loader import load_embedding_model
# from services.embeder import Embeder
# from services.vector_store import VectorStore
# from services.retriever import Retriever
#
#
# def load_csv_texts(csv_path, num_texts=15):
#     """Загрузка текстов из CSV файла"""
#     texts = []
#     try:
#         with open(csv_path, 'r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for i, row in enumerate(reader):
#                 if i >= num_texts:
#                     break
#                 # Предполагаем, что текст в колонке 'text' или первой колонке
#                 text = row.get('text', list(row.values())[0] if row else '')
#                 if text.strip():
#                     texts.append(text.strip())
#         print(f"✅ Загружено {len(texts)} текстов из CSV")
#         return texts
#     except Exception as e:
#         print(f"❌ Ошибка загрузки CSV: {e}")
#         return []
#
#
# def main():
#     # 1. Подключаемся к БД
#     db.connect()
#     db.init_tables()
#
#     # 2. Загружаем модель
#     model = load_embedding_model()
#
#     # 3. Инициализируем сервисы
#     store = VectorStore()
#     embeder = Embeder(model, store)
#     retriever = Retriever()
#
#     # 4. Загружаем тексты из CSV (предположим, файл называется texts.csv)
#     csv_file = data_w = pd.read_csv('websites_updated.csv')['text']  # Укажите правильный путь к вашему CSV
#     texts = load_csv_texts(csv_file, 15)
#
#     if not texts:
#         print("❌ Не удалось загрузить тексты из CSV")
#         return
#
#     # 5. Создаем эмбеддинги и сохраняем в БД
#     print("🔄 Создаем эмбеддинги и сохраняем в БД...")
#     document_ids = embeder.batch_embed_and_store(texts)
#     print(f"✅ Сохранено {len(document_ids)} документов в БД")
#
#     # 6. Тестируем поиск
#     print("\n🔍 Тестируем поиск...")
#     test_queries = list(pd.read_csv('questions_clean.csv')['query'])
#     # test_queries = [
#     #     "основные концепции",  # пример запроса
#     #     "технологии",
#     #     "методы анализа"
#     # ]
#
#     for query in test_queries:
#         print(f"\nПоиск по запросу: '{query}'")
#         try:
#             # Создаем эмбеддинг для запроса
#             query_embedding = embeder.generate_embedding(query)
#
#             # Ищем похожие документы
#             results = retriever.retrieve(query_embedding, limit=5)
#
#             print(f"Найдено результатов: {len(results)}")
#             for i, result in enumerate(results):
#                 print(f"  {i + 1}. Сходство: {result['similarity']:.4f}")
#                 print(f"     Текст: {result['content'][:100]}...")
#
#         except Exception as e:
#             print(f"❌ Ошибка при поиске: {e}")
#
#     # 7. Закрываем соединение
#     db.disconnect()
#     print("\n✅ Задание выполнено!")
#
#
# if __name__ == "__main__":
#     main()

import csv
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, '/app/src')

from config.database import db
from services.model_loader import load_embedding_model
from services.embeder import Embeder
from services.vector_store import VectorStore
from services.retriever import Retriever


def load_csv_texts(csv_path, num_texts=15):
    """Загрузка текстов из CSV файла"""
    texts = []
    try:
        # Используем pandas для чтения CSV
        df = pd.read_csv(csv_path)

        # Предполагаем, что текст в колонке 'text'
        # Если колонки 'text' нет, берем первую колонку
        if 'text' in df.columns:
            text_column = 'text'
        else:
            text_column = df.columns[0]

        # Берем указанное количество текстов
        texts = df[text_column].head(num_texts).dropna().tolist()
        texts = [str(text).strip() for text in texts if str(text).strip()]

        print(f"✅ Загружено {len(texts)} текстов из CSV")
        return texts
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")
        return []


def main():
    # 1. Подключаемся к БД
    db.connect()
    # ОЧИСТКА БД ОТ СТАРЫХ ДАННЫХ
    with db.connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE text_embeddings RESTART IDENTITY;")
        db.connection.commit()
    print("🗑️  Старые данные удалены из БД")
    db.init_tables()

    # 2. Загружаем модель
    model = load_embedding_model()

    # 3. Инициализируем сервисы
    store = VectorStore()
    embeder = Embeder(model, store)
    retriever = Retriever()

    # 4. Загружаем тексты из CSV
    csv_file = 'websites_updated.csv'  # Просто путь к файлу
    texts = load_csv_texts(csv_file, 15)

    if not texts:
        print("❌ Не удалось загрузить тексты из CSV")
        return

    # 5. Создаем эмбеддинги и сохраняем в БД
    print("🔄 Создаем эмбеддинги и сохраняем в БД...")
    document_ids = []
    for i, text in enumerate(texts, 1):
        try:
            doc_id = embeder.embed_and_store(text, {"source": "websites", "index": i})
            if doc_id:
                document_ids.append(doc_id)
                print(f"✅ Текст {i} сохранен с ID: {doc_id}")
        except Exception as e:
            print(f"❌ Ошибка сохранения текста {i}: {e}")

    print(f"✅ Сохранено {len(document_ids)} документов в БД")

    # 6. Тестируем поиск
    print("\n🔍 Тестируем поиск...")

    # Загружаем вопросы из другого CSV
    try:
        questions_df = pd.read_csv('questions_clean.csv')
        test_queries = questions_df['query'].head(3).tolist()  # Берем первые 3 вопроса
    except Exception as e:
        print(f"❌ Ошибка загрузки questions_clean.csv: {e}")
        # Fallback - используем тестовые запросы
        test_queries = [
            "основные концепции",
            "технологии",
            "методы анализа"
        ]

    for query in test_queries:
        print(f"\nПоиск по запросу: '{query}'")
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = embeder.generate_embedding(query)

            # Ищем похожие документы
            results = retriever.retrieve(query_embedding, limit=5)

            print(f"Найдено результатов: {len(results)}")
            for i, result in enumerate(results):
                print(f"  {i + 1}. Сходство: {result['similarity']:.4f}")
                print(f"     Текст: {result['content'][:100]}...")

        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")

    # 7. Закрываем соединение
    db.disconnect()
    print("\n✅ Задание выполнено!")


if __name__ == "__main__":
    main()
