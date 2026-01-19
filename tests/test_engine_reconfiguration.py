import json
from typing import Generator

import pytest

from coreason_search.config import Settings
from coreason_search.db import DocumentSchema, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.engine import SearchEngine
from coreason_search.schemas import RetrieverType, SearchRequest


class TestEngineReconfiguration:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def reset_singletons(self) -> Generator[None, None, None]:
        reset_db_manager()
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def test_multiple_engines_isolation(self, tmp_path: str) -> None:
        """
        Test that two engines initialized with different configurations
        maintain isolation and write to their respective databases,
        despite the global singleton pattern used in factories.
        """
        path_a = str(tmp_path) + "/db_a"
        path_b = str(tmp_path) + "/db_b"

        # Init Engine A
        config_a = Settings(database_uri=path_a)
        engine_a = SearchEngine(config_a)

        # Init Engine B
        config_b = Settings(database_uri=path_b)
        engine_b = SearchEngine(config_b)

        # Seed DB A via Engine A (using its internal components)
        # Note: Engine A's components (DenseRetriever) hold a reference to the
        # DB Manager instance that was current when Engine A was created.

        # We need to manually add data using the managers held by the retrievers
        # because engine doesn't expose "add" method directly.

        embedder = engine_a.embedder  # Should be generic/mock

        # Add doc to A
        table_a = engine_a.dense_retriever.table
        doc_a = DocumentSchema(
            doc_id="doc_a", vector=embedder.embed("A")[0], content="Content A", metadata=json.dumps({"source": "A"})
        )
        table_a.add([doc_a])

        # Add doc to B
        table_b = engine_b.dense_retriever.table
        doc_b = DocumentSchema(
            doc_id="doc_b", vector=embedder.embed("B")[0], content="Content B", metadata=json.dumps({"source": "B"})
        )
        table_b.add([doc_b])

        # Verify Search A returns only A
        req_a = SearchRequest(query="A", strategies=[RetrieverType.LANCE_DENSE])
        res_a = engine_a.execute(req_a)

        # Depending on mock embedding, "A" might match "B" if random vector is close,
        # but we check doc_id presence.
        ids_a = [h.doc_id for h in res_a.hits]
        assert "doc_a" in ids_a
        assert "doc_b" not in ids_a

        # Verify Search B returns only B
        req_b = SearchRequest(query="B", strategies=[RetrieverType.LANCE_DENSE])
        res_b = engine_b.execute(req_b)
        ids_b = [h.doc_id for h in res_b.hits]
        assert "doc_b" in ids_b
        assert "doc_a" not in ids_b

        # Verify physical isolation (files)
        # Check that path_a has data and path_b has data
        # (LanceDB creates files)
        import os

        assert os.path.exists(path_a)
        assert os.path.exists(path_b)
