import os
import tempfile
from unittest.mock import MagicMock

import pytest

from coreason_search.config import Settings, YamlConfigSettingsSource, load_config


class TestConfigLoader:
    def test_load_config_defaults(self) -> None:
        """Test loading defaults when no file exists."""
        # Ensure we don't accidentally pick up a real file
        with pytest.MonkeyPatch.context() as m:
            m.setenv("SEARCH_CONFIG_PATH", "non_existent_file.yaml")
            config = load_config("non_existent_file.yaml")
            assert isinstance(config, Settings)
            assert config.database_uri == "/tmp/lancedb"
            assert config.embedding.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"

    def test_load_config_from_file_and_env_override(self) -> None:
        """Test loading from YAML and overriding with Env vars."""
        config_data = """
database_uri: /tmp/yaml_db
embedding:
  model_name: yaml-model
  context_length: 1024
scout:
  threshold: 0.8
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            temp_path = f.name

        try:
            # Case 1: Load from file only
            config = load_config(temp_path)
            assert config.database_uri == "/tmp/yaml_db"
            assert config.embedding.model_name == "yaml-model"

            # Case 2: Override with Env Var
            with pytest.MonkeyPatch.context() as m:
                m.setenv("APP__DATABASE_URI", "/tmp/env_db")
                m.setenv("APP__EMBEDDING__MODEL_NAME", "env-model")
                # We need to re-load.
                # Note: `load_config` sets SEARCH_CONFIG_PATH env var.
                # Pydantic reads env vars dynamically.
                config_env = load_config(temp_path)

                assert config_env.database_uri == "/tmp/env_db"
                assert config_env.embedding.model_name == "env-model"
                assert config_env.scout.threshold == 0.8  # From YAML

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_settings_customise_sources_coverage(self) -> None:
        """
        Explicitly test settings_customise_sources to ensure 100% coverage.
        This verifies that our custom YAML source is injected.
        """
        init_mock = MagicMock()
        env_mock = MagicMock()
        dotenv_mock = MagicMock()
        file_mock = MagicMock()

        sources = Settings.settings_customise_sources(
            Settings,
            init_settings=init_mock,
            env_settings=env_mock,
            dotenv_settings=dotenv_mock,
            file_secret_settings=file_mock,
        )

        assert len(sources) == 3
        assert sources[0] is init_mock
        assert sources[1] is env_mock
        assert isinstance(sources[2], YamlConfigSettingsSource)
