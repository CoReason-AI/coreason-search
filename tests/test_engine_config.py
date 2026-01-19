import os
import tempfile
from typing import Generator, cast

import pytest
import yaml

from coreason_search.config import EmbeddingConfig, ScoutConfig, Settings
from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.embedders.mock import MockEmbedder
from coreason_search.engine import SearchEngine
from coreason_search.reranker import reset_reranker
from coreason_search.scout import MockScout, reset_scout


class TestEngineConfig:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def reset_singletons(self) -> Generator[None, None, None]:
        reset_db_manager()
        reset_embedder()
        reset_scout()
        reset_reranker()
        yield
        reset_db_manager()
        reset_embedder()
        reset_scout()
        reset_reranker()

    def test_engine_initialization_with_config_object(self) -> None:
        """Test initializing engine with Settings object."""
        config = Settings(database_uri="/tmp/test_engine_db", scout=ScoutConfig(threshold=0.9))
        engine = SearchEngine(config)

        # Verify components got the config
        assert engine.config.database_uri == "/tmp/test_engine_db"
        assert engine.db_manager.uri == "/tmp/test_engine_db"

        # Check Scout config
        scout = cast(MockScout, engine.scout)
        assert scout.config.threshold == 0.9

    def test_engine_initialization_with_yaml_file(self) -> None:
        """Test initializing engine with path to YAML."""
        config_data = {"database_uri": "/tmp/test_yaml_db", "scout": {"threshold": 0.1}}

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            engine = SearchEngine(temp_path)
            assert engine.config.database_uri == "/tmp/test_yaml_db"
            scout = cast(MockScout, engine.scout)
            assert scout.config.threshold == 0.1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_engine_component_propagation(self) -> None:
        """Verify that factories use the config."""
        config = Settings(database_uri="/tmp/prop_db", embedding=EmbeddingConfig(model_name="prop-model"))
        engine = SearchEngine(config)

        # Check Embedder
        embedder = cast(MockEmbedder, engine.embedder)
        assert embedder.config.model_name == "prop-model"

        # Check DB
        assert get_db_manager().uri == "/tmp/prop_db"
