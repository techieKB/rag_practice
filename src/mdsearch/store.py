import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

from mdsearch.embedding import EMBED_DIM
from mdsearch.models import Chunk, SearchHit

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
  id          BIGSERIAL PRIMARY KEY,
  file_path   TEXT NOT NULL,
  file_mtime  TIMESTAMPTZ NOT NULL,
  title       TEXT NOT NULL,
  heading     TEXT NOT NULL,
  chunk_index INT  NOT NULL,
  content     TEXT NOT NULL,
  embedding   vector({EMBED_DIM}) NOT NULL,
  UNIQUE (file_path, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
"""

HIT_COLUMNS = "id, file_path, title, heading, content"


def _vec(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class ChunkStore:
    """Postgres + pgvector storage for Markdown chunks. Safe to share across threads."""

    def __init__(self, dsn: str):
        self._pool = ThreadedConnectionPool(1, 4, dsn)
        self._registered: set[int] = set()

    @contextmanager
    def _conn(self) -> Iterator[connection]:
        conn = self._pool.getconn()
        try:
            if id(conn) not in self._registered:
                # register_vector needs the extension to exist, so create it on first use of each connection.
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                register_vector(conn)
                self._registered.add(id(conn))
            with conn:  # commit on success, rollback on error
                yield conn
        finally:
            self._pool.putconn(conn)

    def init_schema(self) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)

    def file_mtimes(self, root: Path) -> dict[str, datetime]:
        prefix = str(root.resolve()).rstrip(os.sep) + os.sep
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT file_path, file_mtime FROM chunks WHERE starts_with(file_path, %s)",
                (prefix,),
            )
            return dict(cur.fetchall())

    def replace_file(
        self,
        file_path: str,
        mtime: datetime,
        title: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        rows = [
            (file_path, mtime, title, c.heading, c.index, c.content, _vec(e))
            for c, e in zip(chunks, embeddings)
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE file_path = %s", (file_path,))
            execute_values(
                cur,
                "INSERT INTO chunks (file_path, file_mtime, title, heading, chunk_index, content, embedding) VALUES %s",
                rows,
            )

    def delete_files(self, paths: Iterable[str]) -> int:
        paths = list(paths)
        if not paths:
            return 0
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE file_path = ANY(%s)", (paths,))
            return cur.rowcount

    def dense_search(self, query_vec: list[float], limit: int) -> list[SearchHit]:
        q = _vec(query_vec)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {HIT_COLUMNS}, 1 - (embedding <=> %s) FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (q, q, limit),
            )
            return [SearchHit(*row[:5], score=float(row[5])) for row in cur.fetchall()]

    def all_chunks(self) -> list[SearchHit]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {HIT_COLUMNS} FROM chunks ORDER BY id")
            return [SearchHit(*row, score=0.0) for row in cur.fetchall()]

    def signature(self) -> tuple[int, int]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), coalesce(max(id), 0) FROM chunks")
            count, max_id = cur.fetchone()
            return int(count), int(max_id)

    def truncate(self) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE chunks RESTART IDENTITY")

    def close(self) -> None:
        self._pool.closeall()
