# config.py
# Совместимо с Python 3.10 и Pydantic v2 / pydantic-settings v2

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Единая конфигурация приложения.
    - Значения читаются из переменных окружения и файла .env (если он есть).
    - Безопасно работать без GENAPI_KEY (например, в тестах); реальный вызов API
      должен сам проверить наличие ключа и отреагировать.
    """

    # Ключ для GenAPI. Можно не задавать в тестах/локалке.
    genapi_key: Optional[str] = None

    # Базовый URL сети GenAPI (по умолчанию gpt-4o-сеть)
    genapi_url: str = Field(
        default="https://api.gen-api.ru/api/v1/networks/gpt-4o"
    )

    # Переменные окружения: INDEX_PATH, META_PATH, BM25_PATH
    index_path: str = Field(default="./faq.index")
    meta_path: str = Field(default="./faq_meta.pkl")
    bm25_path: str = Field(default="./bm25.pkl")

    # Доля dense-скоринга: 1.0 — только FAISS, 0.0 — только BM25
    # Переменная окружения: HYBRID_ALPHA
    hybrid_alpha: float = Field(default=0.6)

    # Сколько кандидатов забираем из FAISS/BM25 для смешивания
    # Переменная окружения: FAISS_K
    faiss_k: int = Field(default=50)

    # Сколько финальных фрагментов отдаём генератору
    # Переменная окружения: TOP_K
    top_k: int = Field(default=5)

    # === Сервисные параметры ===
    # Таймаут HTTP-запроса к GenAPI (сек)
    # Переменная окружения: REQUEST_TIMEOUT_SEC
    request_timeout_sec: int = Field(default=60)

    # Ограничения на объём контекста (промпт-оптимизация)
    # Переменные окружения: MAX_FRAGMENT_CHARS, MAX_CONTEXT_CHARS
    # max_fragment_chars — обрезка одного фрагмента
    # max_context_chars — общее ограничение контекста, которое возвращаем в API/модель
    max_fragment_chars: int = Field(default=800)
    max_context_chars: int = Field(default=600)

    # Настройки загрузки из .env, игнор лишних переменных, нечувствительность к порядку
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


# Глобальный объект настроек
settings = Settings()
