import pandas as pd
import sys
import os

sys.path.append('/app/src')

from config.database import db
from src.services.model_loader import load_embedding_model
from src.services.embeder import Embeder
from src.services.retriever import Retriever


def generate_proper_submission():
    try:
        print("🎯 Генерация submission.csv с использованием модели...")

        # 1. Подключаемся к БД
        db.connect()

        # 2. Загружаем модель и сервисы
        model = load_embedding_model()
        embeder = Embeder(model, store=None)
        retriever = Retriever()

        # 3. Загружаем вопросы
        questions_df = pd.read_csv('questions_clean.csv')
        print(f"✅ Загружено {len(questions_df)} вопросов")

        # 4. Для каждого вопроса находим 5 самых релевантных документов
        results = []

        for index, row in questions_df.iterrows():
            q_id = index + 1
            query = row['query']

            print(f"\n🔍 Обрабатываем вопрос {q_id}: '{query}'")

            try:
                # Создаем эмбеддинг для запроса
                query_embedding = embeder.generate_embedding(query)

                # Ищем 5 самых похожих документов
                search_results = retriever.retrieve(query_embedding, limit=5)

                if search_results:
                    # Извлекаем ID найденных документов
                    doc_ids = [result['id'] for result in search_results]

                    # Форматируем в нужный формат
                    web_list_str = "[" + ", ".join(map(str, doc_ids)) + "]"

                    results.append({
                        'q_id': q_id,
                        'web_list': web_list_str
                    })

                    print(f"   ✅ Найдено документов: {len(doc_ids)}")
                    print(f"   📋 ID документов: {web_list_str}")

                    # Покажем сходства для отладки
                    for i, result in enumerate(search_results):
                        print(f"      {i + 1}. ID {result['id']}, сходство: {result['similarity']:.4f}")
                        print(f"         {result['content'][:80]}...")
                else:
                    print(f"   ⚠️  Не найдено документов для вопроса {q_id}")
                    # Fallback - пустой список
                    results.append({
                        'q_id': q_id,
                        'web_list': "[]"
                    })

            except Exception as e:
                print(f"   ❌ Ошибка при обработке вопроса {q_id}: {e}")
                # Fallback - пустой список при ошибке
                results.append({
                    'q_id': q_id,
                    'web_list': "[]"
                })

        # 5. Сохраняем в CSV
        submission_df = pd.DataFrame(results)
        submission_df.to_csv('submission.csv', index=False)

        print(f"\n🎉 Файл submission.csv создан!")
        print(f"📊 Обработано вопросов: {len(results)}")
        print(f"📁 Сохранено в: /app/submission.csv")

        # Статистика
        successful_queries = sum(1 for r in results if r['web_list'] != "[]")
        print(f"📈 Успешно обработано запросов: {successful_queries}/{len(results)}")

        # Покажем первые несколько строк
        print("\n📄 Содержимое submission.csv:")
        print(submission_df.head(10))

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db.connection:
            db.disconnect()


if __name__ == "__main__":
    generate_proper_submission()