import math
import re
import zlib

from mdsearch.embedding import EMBED_DIM


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: texts sharing words score high cosine similarity."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode()) % EMBED_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [x / norm for x in vec]
