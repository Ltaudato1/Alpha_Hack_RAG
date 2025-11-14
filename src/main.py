from config.database import db
from psycopg2.extras import Json

def test_docker_database():
    try:
        print("🔌 Тестируем подключение к Docker PostgreSQL...")
        
        db.connect()
        
        db.init_tables()
        
        with db.connection.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'text_embeddings'
            """)
            table_exists = cursor.fetchone()
            print(f"✅ Таблица text_embeddings: {'существует' if table_exists else 'не найдена'}")
            
            # Проверяем расширение vector
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            vector_exists = cursor.fetchone()['exists']
            print(f"✅ Расширение vector: {'установлено' if vector_exists else 'не установлено'}")
            
            # Пробуем вставить тестовые данные
            cursor.execute("""
                INSERT INTO text_embeddings (text_content, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id
            """, ("Тестовый текст из Docker", [0.1] * 768, Json({"source": "docker_test"})))
            
            inserted_id = cursor.fetchone()['id']
            db.connection.commit()
            print(f"✅ Тестовые данные добавлены с ID: {inserted_id}")
            
            # Проверяем что данные есть
            cursor.execute("SELECT COUNT(*) as count FROM text_embeddings")
            count = cursor.fetchone()['count']
            print(f"✅ Всего записей в таблице: {count}")
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    test_docker_database()