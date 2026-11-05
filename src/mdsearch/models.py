from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    heading: str  # heading path, e.g. "Title > Section > Subsection"
    content: str
    index: int


@dataclass(frozen=True)
class ParsedDoc:
    title: str
    chunks: list[Chunk]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    file_path: str
    title: str
    heading: str
    content: str
    score: float
