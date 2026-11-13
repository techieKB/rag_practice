from datetime import UTC, datetime
from pathlib import Path

import pytest

from fakes import FakeEmbedder
from mdsearch.models import Chunk

pytestmark = pytest.mark.integration

MTIME = datetime(2026, 1, 1, tzinfo=UTC)
FE = FakeEmbedder()


def _store_file(store, path: str, contents: list[str]) -> None:
    chunks = [Chunk(f"T > S{i}", c, i) for i, c in enumerate(contents)]
    store.replace_file(path, MTIME, "T", chunks, FE.embed_documents(contents))


def test_replace_file_then_dense_search_ranks_nearest_first(store):
    chunks = [Chunk("T > Postgres", "gin index for full text", 0), Chunk("T > SSH", "proxyjump bastion", 1)]
    store.replace_file("/n/a.md", MTIME, "T", chunks, FE.embed_documents([c.content for c in chunks]))
    hits = store.dense_search(FE.embed_query("gin index"), limit=2)
    assert hits[0].heading == "T > Postgres" and hits[0].file_path == "/n/a.md"
    assert hits[0].title == "T" and hits[0].content == "gin index for full text"
    assert hits[0].score > hits[1].score


def test_replace_file_is_idempotent(store):
    _store_file(store, "/n/a.md", ["one", "two"])
    _store_file(store, "/n/a.md", ["one", "two"])
    assert store.signature()[0] == 2


def test_delete_files_returns_count(store):
    _store_file(store, "/n/a.md", ["one", "two"])
    _store_file(store, "/n/b.md", ["three"])
    assert store.delete_files(["/n/a.md"]) == 2
    assert [h.file_path for h in store.all_chunks()] == ["/n/b.md"]


def test_file_mtimes_scoped_to_root(store):
    _store_file(store, "/n/a.md", ["one"])
    _store_file(store, "/other/b.md", ["two"])
    _store_file(store, "/nx/c.md", ["three"])
    assert store.file_mtimes(Path("/n")) == {"/n/a.md": MTIME}


def test_signature_changes_on_replace(store):
    s0 = store.signature()
    _store_file(store, "/n/a.md", ["one"])
    assert store.signature() != s0
