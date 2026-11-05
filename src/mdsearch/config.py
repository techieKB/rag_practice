import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    test_database_url: str
    ollama_host: str
    embed_model: str
    rerank_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ.get(
                "MDSEARCH_DATABASE_URL", "postgresql://mdsearch:mdsearch@localhost:5433/mdsearch"
            ),
            test_database_url=os.environ.get(
                "MDSEARCH_TEST_DATABASE_URL", "postgresql://mdsearch:mdsearch@localhost:5433/mdsearch_test"
            ),
            ollama_host=os.environ.get("MDSEARCH_OLLAMA_HOST", "http://localhost:11434"),
            embed_model=os.environ.get("MDSEARCH_EMBED_MODEL", "nomic-embed-text"),
            rerank_model=os.environ.get("MDSEARCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        )
