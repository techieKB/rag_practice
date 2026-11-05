import math
from unittest.mock import Mock

import pytest

from fakes import FakeEmbedder
from mdsearch.config import Settings
from mdsearch.embedding import EMBED_DIM, OllamaEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_documents_are_prefixed_and_batched():
    client = Mock()
    client.embed.side_effect = lambda model, input: {"embeddings": [[0.0] * EMBED_DIM for _ in input]}
    emb = OllamaEmbedder("http://x", "nomic-embed-text", batch_size=2, client=client)
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and client.embed.call_count == 2
    assert client.embed.call_args_list[0].kwargs["input"] == ["search_document: a", "search_document: b"]
    assert client.embed.call_args_list[1].kwargs["input"] == ["search_document: c"]


def test_query_is_prefixed():
    client = Mock()
    client.embed.return_value = {"embeddings": [[0.0] * EMBED_DIM]}
    OllamaEmbedder("http://x", "m", client=client).embed_query("hi")
    assert client.embed.call_args.kwargs["input"] == ["search_query: hi"]


def test_wrong_dimension_raises():
    client = Mock()
    client.embed.return_value = {"embeddings": [[0.0] * 10]}
    with pytest.raises(ValueError, match="768"):
        OllamaEmbedder("http://x", "m", client=client).embed_query("hi")


def test_empty_document_list_makes_no_call():
    client = Mock()
    assert OllamaEmbedder("http://x", "m", client=client).embed_documents([]) == []
    client.embed.assert_not_called()


def test_fake_embedder_similar_texts_closer():
    fe = FakeEmbedder()
    base = fe.embed_query("postgres index")
    assert _cosine(base, fe.embed_query("postgres gin index")) > _cosine(base, fe.embed_query("ssh config"))
    assert len(base) == EMBED_DIM


@pytest.mark.integration
def test_live_ollama_returns_768_dims():
    s = Settings.from_env()
    assert len(OllamaEmbedder(s.ollama_host, s.embed_model).embed_query("hello")) == EMBED_DIM
