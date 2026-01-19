import os
import tempfile

import pytest
import yaml

from coreason_search.config import load_config


class TestConfigEdgeCases:
    def test_load_config_partial(self) -> None:
        """Test loading a YAML with only some sections."""
        config_data = {"database_uri": "/tmp/partial_db"}
        # embedding, reranker, scout missing

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.database_uri == "/tmp/partial_db"
            # Verify defaults for missing sections
            assert config.embedding.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"
            assert config.scout.threshold == 0.4
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_empty_file(self) -> None:
        """Test loading a completely empty YAML file."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Should be all defaults
            assert config.database_uri == "/tmp/lancedb"
            assert config.embedding.context_length == 32768
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_type_mismatch(self) -> None:
        """Test validation error when types don't match."""
        config_data = {"embedding": {"context_length": "not-an-integer"}}

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid configuration file"):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_extra_fields(self) -> None:
        """Test that extra fields are ignored (default Pydantic behavior) or handled."""
        config_data = {
            "database_uri": "/tmp/extra_db",
            "unknown_section": {"foo": "bar"},
            "embedding": {"unknown_field": "baz"},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # By default Pydantic V2 ignores extra fields unless ConfigDict(extra='forbid')
            # Our config does NOT set extra='forbid', so it should succeed.
            config = load_config(temp_path)
            assert config.database_uri == "/tmp/extra_db"
            # Ensure it didn't crash
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
