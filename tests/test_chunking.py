from mdsearch.chunking import iter_markdown_files, parse_markdown
from mdsearch.models import Chunk


def test_splits_on_h1_h2_h3_and_builds_heading_path():
    doc = parse_markdown("# Title\nintro\n## A\na body\n### A1\na1 body\n## B\nb body\n", "fallback")
    assert doc.title == "Title"
    assert [c.heading for c in doc.chunks] == ["Title", "Title > A", "Title > A > A1", "Title > B"]
    assert doc.chunks[1].content == "a body"
    assert [c.index for c in doc.chunks] == [0, 1, 2, 3]


def test_h4_does_not_split():
    doc = parse_markdown("## A\nx\n#### Deep\ny\n", "f")
    assert len(doc.chunks) == 1 and "#### Deep" in doc.chunks[0].content


def test_hash_lines_inside_fenced_code_do_not_split():
    text = "## Setup\n```bash\n# install deps\npip install x\n```\n~~~python\n## not a header\n~~~\nafter\n"
    doc = parse_markdown(text, "f")
    assert len(doc.chunks) == 1
    assert "# install deps" in doc.chunks[0].content and "after" in doc.chunks[0].content


def test_no_headers_gives_single_chunk_titled_by_fallback():
    doc = parse_markdown("just some text\n", "my-note")
    assert doc.title == "my-note"
    assert doc.chunks == [Chunk(heading="my-note", content="just some text", index=0)]


def test_front_matter_stripped_and_empty_sections_dropped():
    doc = parse_markdown("---\ntags: [x]\n---\n# T\n## Empty\n## Full\nbody\n", "f")
    assert [c.heading for c in doc.chunks] == ["T > Full"]
    assert all("tags" not in c.content for c in doc.chunks)


def test_hash_without_space_is_not_header():
    assert len(parse_markdown("## A\n#hashtag text\n", "f").chunks) == 1


def test_empty_file_gives_no_chunks():
    assert parse_markdown("", "f").chunks == []


def test_iter_markdown_files_skips_hidden_and_non_md(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("x")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "d.md").write_text("x")
    assert [p.relative_to(tmp_path).as_posix() for p in iter_markdown_files(tmp_path)] == ["a.md", "sub/c.md"]
