"""
Centralized Configuration Module using Pydantic Settings.
Loads configuration from environment variables or .env files.
Extended for Phase 3 Multi-Agent LangGraph System and PostgreSQL Conversational Memory.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Settings
    APP_ENV: str = "development"
    APP_NAME: str = "Multi-Agent Telugu Conversational AI"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    SECRET_KEY: str = "development-secret-key-change-in-production"

    # PostgreSQL Relational Database (Conversational Memory)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "telugu_ai_db"
    POSTGRES_URL: Optional[str] = None

    @property
    def database_url(self) -> str:
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "telugu_documents"

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32

    # Retrieval & Ranking Settings (Phase 2 Advanced RAG)
    RETRIEVAL_MODE: str = "hybrid"  # "dense", "bm25", "hybrid", "hybrid_rerank"
    RETRIEVAL_TOP_K: int = 4
    MIN_SIMILARITY_SCORE: float = 0.35
    
    # BM25 Parameters
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    
    # Reciprocal Rank Fusion (RRF) Parameters
    RRF_K: int = 60
    
    # Cross-Encoder Reranker Parameters
    RERANKER_PROVIDER: str = "local"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_TOP_N: int = 10

    # Multi-Agent & LangGraph Orchestration (Phase 3)
    MAX_VALIDATION_RETRIES: int = 2
    CONVERSATION_HISTORY_LIMIT: int = 6
    DEFAULT_RETRIEVAL_MODE: str = "hybrid"

    # LLM Settings
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.6-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048

    # API Keys & URLs
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Chunking Parameters
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120

    # Speech-to-Text (STT) Settings (Phase 5)
    STT_ENGINE: str = "whisper_turbo"  # "whisper_turbo", "indicconformer", "mock"
    STT_MODEL_NAME: str = "openai/whisper-large-v3-turbo"
    STT_LANGUAGE: Optional[str] = None  # None / "auto" enables Whisper auto-detection for Telugu & English
    STT_DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Text-to-Speech (TTS) Settings (Phase 5 & 7)
    TTS_ENGINE: str = "indicf5"  # "indicf5", "fallback"
    TTS_MODEL_NAME: str = "ai4bharat/IndicF5"
    TTS_SAMPLE_RATE: int = 24000
    TTS_DEFAULT_VOICE: str = "telugu_female_1"
    TTS_REFERENCE_AUDIO_PATH: Optional[str] = None
    TTS_ADAPTER: str = "models/indicf5_telugu_lora"  # "none" or path to adapter folder, e.g. "models/indicf5_telugu_lora"
    TTS_DEVICE: str = "auto"  # "auto", "cpu", "cuda", "mps"


# Global settings instance
settings = Settings()

