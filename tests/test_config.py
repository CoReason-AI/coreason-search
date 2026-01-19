import os
import tempfile

import pytest
import yaml

from coreason_search.config import load_config
from coreason_search.schemas import AppConfig


class TestConfigLoader:
    def test_load_config_defaults(self) -> None:
        """Test loading defaults when no file exists."""
        # Ensure we don't accidentally pick up a real file
        with pytest.MonkeyPatch.context() as m:
            m.setenv("SEARCH_CONFIG_PATH", "non_existent_file.yaml")
            config = load_config("non_existent_file.yaml")
            assert isinstance(config, AppConfig)
            assert config.database_uri == "/tmp/lancedb"
            assert config.embedding.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"

    def test_load_config_from_file(self) -> None:
        """Test loading from a valid YAML file."""
        config_data = {
            "database_uri": "/tmp/custom_db",
            "embedding": {"model_name": "custom-model", "context_length": 1024},
            "scout": {"threshold": 0.8},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.database_uri == "/tmp/custom_db"
            assert config.embedding.model_name == "custom-model"
            assert config.embedding.context_length == 1024
            assert config.scout.threshold == 0.8
            # Check default remains for unspecified
            assert config.embedding.batch_size == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_invalid_yaml(self) -> None:
        """Test error handling for invalid YAML."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: : content")
            temp_path = f.name

        try:
            # Pydantic validation error or YAML error
            with pytest.raises(ValueError):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
