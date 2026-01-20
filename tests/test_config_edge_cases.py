import os
import tempfile

import pytest
from pydantic import ValidationError

from coreason_search.config import load_config


class TestConfigEdgeCases:
    def test_load_config_partial(self) -> None:
        """Test loading a YAML with only some sections."""
        config_data = """
database_uri: /tmp/partial_db
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
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

    def test_load_config_syntax_error(self) -> None:
        """Test loading invalid YAML syntax."""
        config_data = """
database_uri: /tmp/db
key: : value  # Syntax error
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            temp_path = f.name

        try:
            # Should raise ValueError as per our YamlConfigSettingsSource wrapper
            with pytest.raises(ValueError, match="Invalid configuration file"):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_type_mismatch(self) -> None:
        """Test validation error when types don't match."""
        config_data = """
embedding:
  context_length: not-an-integer
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_config_extra_fields(self) -> None:
        """Test that extra fields are ignored (extra='ignore')."""
        config_data = """
database_uri: /tmp/extra_db
unknown_section:
  foo: bar
embedding:
  unknown_field: baz
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_data)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.database_uri == "/tmp/extra_db"
            # Ensure it didn't crash
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
