import os

import pytest

from fakes import FakeEmbedder
from mdsearch.ingest import embedding_text, ingest_directory, main


class SpyEmbedder(FakeEmbedder):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def _p(path) -> str:
    return str(path.resolve())


def _bump_mtime(path, seconds: int = 10) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + seconds))


def test_embedding_text_prefixes_title_only_when_missing():
    assert embedding_text("T", "T > A", "body") == "T > A\n\nbody"
    assert embedding_text("note", "A", "body") == "note > A\n\nbody"
    assert embedding_text("Py", "Python > X", "body") == "Py > Python > X\n\nbody"


def test_main_rejects_missing_dir():
    assert main(["/does/not/exist"]) == 2


@pytest.mark.integration
def test_ingests_all_files_with_metadata(tmp_path, store):
    (tmp_path / "a.md").write_text("# Alpha\n## One\nx\n## Two\ny\n")
    report = ingest_directory(tmp_path, store, FakeEmbedder())
    assert report.indexed == [_p(tmp_path / "a.md")]
    rows = store.all_chunks()
    assert [r.heading for r in rows] == ["Alpha > One", "Alpha > Two"]
    assert rows[0].title == "Alpha"


@pytest.mark.integration
def test_second_run_unchanged_skips_embedding(tmp_path, store):
    (tmp_path / "a.md").write_text("## One\nx\n")
    ingest_directory(tmp_path, store, FakeEmbedder())
    spy = SpyEmbedder()
    report = ingest_directory(tmp_path, store, spy)
    assert report.unchanged == [_p(tmp_path / "a.md")] and report.indexed == []
    assert spy.calls == 0


@pytest.mark.integration
def test_edited_file_replaced_not_duplicated(tmp_path, store):
    note = tmp_path / "a.md"
    note.write_text("## One\nx\n## Two\ny\n")
    ingest_directory(tmp_path, store, FakeEmbedder())
    note.write_text("## Only\nz\n")
    _bump_mtime(note)
    ingest_directory(tmp_path, store, FakeEmbedder())
    rows = store.all_chunks()
    assert [(r.heading, r.content) for r in rows] == [("Only", "z")]


@pytest.mark.integration
def test_deleted_and_renamed_files_removed(tmp_path, store):
    (tmp_path / "a.md").write_text("## A\nx\n")
    (tmp_path / "b.md").write_text("## B\ny\n")
    ingest_directory(tmp_path, store, FakeEmbedder())
    (tmp_path / "a.md").rename(tmp_path / "c.md")
    (tmp_path / "b.md").unlink()
    report = ingest_directory(tmp_path, store, FakeEmbedder())
    assert sorted(report.deleted) == [_p(tmp_path / "a.md"), _p(tmp_path / "b.md")]
    assert {r.file_path for r in store.all_chunks()} == {_p(tmp_path / "c.md")}


@pytest.mark.integration
def test_second_root_does_not_delete_first(tmp_path, store):
    (tmp_path / "root1").mkdir()
    (tmp_path / "root2").mkdir()
    (tmp_path / "root1" / "x.md").write_text("## X\nx\n")
    (tmp_path / "root2" / "y.md").write_text("## Y\ny\n")
    ingest_directory(tmp_path / "root1", store, FakeEmbedder())
    ingest_directory(tmp_path / "root2", store, FakeEmbedder())
    assert {r.file_path for r in store.all_chunks()} == {
        _p(tmp_path / "root1" / "x.md"),
        _p(tmp_path / "root2" / "y.md"),
    }


@pytest.mark.integration
def test_undecodable_file_skipped_others_ingested(tmp_path, store):
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad")
    (tmp_path / "good.md").write_text("## G\ntext\n")
    report = ingest_directory(tmp_path, store, FakeEmbedder())
    assert report.skipped == [_p(tmp_path / "bad.md")]
    assert {r.file_path for r in store.all_chunks()} == {_p(tmp_path / "good.md")}


@pytest.mark.integration
def test_file_emptied_removes_its_chunks(tmp_path, store):
    note = tmp_path / "a.md"
    note.write_text("## A\nx\n")
    ingest_directory(tmp_path, store, FakeEmbedder())
    note.write_text("")
    _bump_mtime(note)
    report = ingest_directory(tmp_path, store, FakeEmbedder())
    assert report.indexed == [_p(note)] and store.all_chunks() == []
