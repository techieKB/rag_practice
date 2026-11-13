import pytest

from mdsearch.config import Settings
from mdsearch.store import ChunkStore


@pytest.fixture
def store():
    s = ChunkStore(Settings.from_env().test_database_url)
    s.init_schema()
    s.truncate()
    yield s
    s.close()
