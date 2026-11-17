import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mdsearch.chunking import iter_markdown_files, parse_markdown
from mdsearch.config import Settings
from mdsearch.embedding import Embedder, OllamaEmbedder
from mdsearch.store import ChunkStore

log = logging.getLogger(__name__)


def embedding_text(title: str, heading: str, content: str) -> str:
    """Text fed to the embedder and BM25: heading path (always starting with the title) plus body."""
    starts_with_title = heading == title or heading.startswith(f"{title} > ")
    path = heading if starts_with_title else f"{title} > {heading}"
    return f"{path}\n\n{content}"


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def ingest_directory(root: Path, store: ChunkStore, embedder: Embedder) -> IngestReport:
    """Bring the store in sync with the .md files under root: embed new/changed files, drop removed ones."""
    root = root.resolve()
    known = store.file_mtimes(root)
    report = IngestReport()
    seen: set[str] = set()

    for path in iter_markdown_files(root):
        key = str(path)
        seen.add(key)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if known.get(key) == mtime:
            report.unchanged.append(key)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            # Keep any previously stored chunks for this file; just don't update them.
            log.warning("skipping %s: %s", key, exc)
            report.skipped.append(key)
            continue

        doc = parse_markdown(text, fallback_title=path.stem)
        if doc.chunks:
            vectors = embedder.embed_documents(
                [embedding_text(doc.title, c.heading, c.content) for c in doc.chunks]
            )
            store.replace_file(key, mtime, doc.title, doc.chunks, vectors)
        else:
            store.delete_files([key])
        report.indexed.append(key)

    report.deleted = sorted(known.keys() - seen)
    store.delete_files(report.deleted)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mdsearch-ingest", description="Index a folder of Markdown notes.")
    parser.add_argument("notes_dir", type=Path)
    args = parser.parse_args(argv)
    if not args.notes_dir.is_dir():
        print(f"error: {args.notes_dir} is not a directory", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    store = ChunkStore(settings.database_url)
    try:
        store.init_schema()
        report = ingest_directory(
            args.notes_dir, store, OllamaEmbedder(settings.ollama_host, settings.embed_model)
        )
    finally:
        store.close()
    print(
        f"indexed={len(report.indexed)} unchanged={len(report.unchanged)} "
        f"deleted={len(report.deleted)} skipped={len(report.skipped)}"
    )
    return 0
