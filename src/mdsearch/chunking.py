import os
import re
from collections.abc import Iterator
from pathlib import Path

from mdsearch.models import Chunk, ParsedDoc

HEADER_RE = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.+?)[ \t#]*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def _strip_front_matter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return lines[i + 1 :]
    return lines


def parse_markdown(text: str, fallback_title: str) -> ParsedDoc:
    """Split a Markdown document into chunks at #, ## and ### headers (outside fenced code)."""
    lines = _strip_front_matter(text.splitlines())

    title: str | None = None
    stack: list[tuple[int, str]] = []  # (level, heading text)
    sections: list[tuple[list[str], list[str]]] = []  # (heading path, body lines)
    body: list[str] = []
    path: list[str] = []
    fence: str | None = None

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            body.append(line)
            continue
        header = None if fence else HEADER_RE.match(line)
        if header is None:
            body.append(line)
            continue
        sections.append((path, body))
        level, heading = len(header.group(1)), header.group(2).strip()
        if level == 1 and title is None:
            title = heading
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        path = [h for _, h in stack]
        body = []
    sections.append((path, body))

    title = title or fallback_title
    chunks: list[Chunk] = []
    for heading_path, body_lines in sections:
        content = "\n".join(body_lines).strip()
        if content:
            heading = " > ".join(heading_path) if heading_path else title
            chunks.append(Chunk(heading=heading, content=content, index=len(chunks)))
    return ParsedDoc(title=title, chunks=chunks)


def iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield .md files under root in sorted order, skipping hidden paths and symlinks."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if name.endswith(".md") and not name.startswith(".") and not path.is_symlink():
                found.append(path)
    yield from sorted(found, key=lambda p: p.relative_to(root).as_posix())
