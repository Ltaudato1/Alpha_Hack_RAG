from src.config.database import db
import pandas as pd
import re
from psycopg2.extras import Json
from src.services.vector_store import VectorStore
from llama_cpp import Llama

from src.services.vector_store import VectorStore
from src.services.embeder import Embeder
from src.services.retriever import Retriever


import os


def load_embedding_model():
    """Загрузка GGUF модели для эмбеддингов"""
    model_path = "models/jina-embeddings-v4-text-retrieval-IQ1_S.gguf"
    #model_path = "models/qodo-embed-1-1.5b-q4_k_m.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    
    model = Llama(
            model_path=model_path,
            embedding=True,
            n_threads=6,
            n_threads_batch=6,
            verbose=False
        )

    print("✅ Модель для эмбеддингов загружена")
    return model

def chunk_text(text, chunk_size=250, overlap=50):
    """
    Разбивает текст на чанки заданного размера с перекрытием
    
    Args:
        text (str): исходный текст
        chunk_size (int): размер чанка в словах
        overlap (int): перекрытие между чанками в словах
    
    Returns:
        list: список текстовых чанков
    """
    if not text or not isinstance(text, str):
        return []
    
    # Очищаем текст от лишних пробелов
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Разбиваем на слова
    words = text.split()
    
    # Если текст короче chunk_size, возвращаем как есть
    if len(words) <= chunk_size:
        return [' '.join(words)] if words else []
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        start += chunk_size - overlap
        
        if start >= len(words):
            break
            
        remaining = len(words) - start
        if remaining < chunk_size and remaining > overlap:
            last_chunk = ' '.join(words[start:])
            chunks.append(last_chunk)
            break
    
    return chunks

def load_csv_texts(csv_path, num_texts=None, chunk_size=250, overlap=50, min_chunk_length=50):
    """Загрузка текстов из CSV файла с разбивкой на чанки"""
    
    texts = []
    idxs = []
    try:
        df = pd.read_csv(csv_path).dropna()
        text_column = 'text'
        id_column = 'web_id'

        if num_texts is None:
            num_texts = len(df)
        else:
            num_texts = min(num_texts, len(df))
        
        raw_texts = df[text_column].head(num_texts).tolist()
        raw_texts = [str(text).strip() for text in raw_texts if str(text).strip()]
        ids = df[id_column].head(num_texts).tolist()
        
        print(f"✅ Загружено {len(raw_texts)} исходных текстов из CSV")
        
        total_chunks = 0
        verbose = 100
        for i, text in enumerate(raw_texts):
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            filtered_chunks = [
                chunk for chunk in chunks 
                if len(chunk.split()) >= min_chunk_length
            ]
            
            texts.extend(filtered_chunks)
            total_chunks += len(filtered_chunks)
            idxs.extend([{'id': ids[i]}] * len(filtered_chunks))
            
            if (i + 1) % verbose == 0:
                print(f"📊 Обработано {i + 1}/{len(raw_texts)} текстов, создано {total_chunks} чанков")
        
        print(f"🎯 Итог: {len(raw_texts)} текстов → {len(texts)} чанков")

        return texts, idxs

        
        
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")
        return [], []

def generate_proper_submission():
    try:
        print("🎯 Генерация submission.csv с использованием модели...")

        model = load_embedding_model()
        embeder = Embeder(model, store=None)
        retriever = Retriever()

        questions_df = pd.read_csv('questions_clean.csv')
        print(f"✅ Загружено {len(questions_df)} вопросов")

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
                    doc_ids = [result['metadata']['id'] for result in search_results]

                    # Форматируем в нужный формат
                    web_list_str = "[" + ", ".join(map(str, doc_ids)) + "]"

                    results.append({
                        'q_id': q_id,
                        'web_list': web_list_str
                    })

                    print(f"   ✅ Найдено документов: {len(doc_ids)}")
                    print(f"   📋 ID документов: {web_list_str}")

                    for i, result in enumerate(search_results):
                        print(f"      {i + 1}. ID {result['metadata']['id']}, сходство: {result['similarity']:.4f}")
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

def clean_db():
    with db.connection.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE text_embeddings RESTART IDENTITY;")
        db.connection.commit()

def create_embeddings():
    clean_db()
    data, metadata = load_csv_texts(csv_path='websites.csv')
    
    store = VectorStore()
    emb_model = load_embedding_model()
    emb = Embeder(emb_model, store)
    emb.batch_embed_and_store(documents=data, metadata_list=metadata)



def main():
    db.connect()
    create_embeddings()
    generate_proper_submission()
    db.disconnect()
    
    

if __name__ == "__main__":
    main()