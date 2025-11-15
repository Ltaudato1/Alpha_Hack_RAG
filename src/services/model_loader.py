from llama_cpp import Llama
import os


def load_embedding_model():
    """Загрузка GGUF модели для эмбеддингов"""
    model_path = "models/all-MiniLM-L6-v2.Q4_0.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    model = Llama(
        model_path=model_path,
        embedding=True,  # Включаем режим эмбеддингов
        n_ctx=2048,  # Размер контекста
        verbose=False
    )

    print("✅ Модель для эмбеддингов загружена")
    return model