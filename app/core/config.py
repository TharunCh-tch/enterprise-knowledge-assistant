from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Knowledge Assistant"
    APP_VERSION: str = "1.0.0"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "google/flan-t5-base"
    LLM_MAX_NEW_TOKENS: int = 256

    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3

    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md"]

    model_config = {"env_file": ".env"}

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def upload_dir(self) -> Path:
        return self.base_dir / "data" / "uploads"

    @property
    def faiss_dir(self) -> Path:
        return self.base_dir / "data" / "faiss_index"


settings = Settings()
