from typing import Protocol

import ollama

EMBED_DIM = 768

# nomic-embed-text expects task prefixes on every input.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, host: str, model: str, batch_size: int = 32, client: ollama.Client | None = None):
        self.model = model
        self.batch_size = batch_size
        self.client = client or ollama.Client(host=host)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed([DOCUMENT_PREFIX + t for t in batch]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors = [list(v) for v in self.client.embed(model=self.model, input=inputs)["embeddings"]]
        for v in vectors:
            if len(v) != EMBED_DIM:
                raise ValueError(f"expected {EMBED_DIM}-dim embedding from {self.model!r}, got {len(v)}")
        return vectors
