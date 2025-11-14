from config.database import db
from psycopg2.extras import Json
from services.vector_store import VectorStore

def test_docker_database():
    try:
        print("🔌 Тестируем подключение к Docker PostgreSQL...")
        
        db.connect()
        
        db.init_tables()

        store = VectorStore()

        store.store_embedding("Текстовый текст из Docker", [0.1] * 768, Json({"source": "docker_test"}))
        
        res = store.get_all_texts()
        
        print(res)

    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        db.disconnect()

if __name__ == "__main__":
    test_docker_database()