# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import os
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from coreason_search.utils.logger import logger


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model.

    Attributes:
        provider: The provider of the embedder ('auto', 'hf', 'mock').
        model_name: The name of the model to use (e.g., HuggingFace model ID).
        context_length: The maximum context length for the embeddings.
        batch_size: The batch size for embedding generation.
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["auto", "hf", "mock"] = Field(
        default="auto", description="Embedder provider: 'auto', 'hf', 'mock'"
    )
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    context_length: int = Field(default=32768, gt=0)
    batch_size: int = Field(default=1, gt=0)


class RerankerConfig(BaseModel):
    """Configuration for the re-ranking model.

    Attributes:
        model_name: The name of the cross-encoder model to use.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"


class ScoutConfig(BaseModel):
    """Configuration for the Scout (Context Distiller).

    Attributes:
        model_name: The name of the model used for distillation.
        threshold: The threshold score for filtering irrelevant segments.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """A settings source that loads configuration from a YAML file.

    This source looks for a YAML file at the path specified by the
    SEARCH_CONFIG_PATH environment variable, defaulting to 'search_config.yaml'.
    """

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        """Gets the value for a specific field.

        Args:
            field: The field to retrieve.
            field_name: The name of the field.

        Returns:
            A tuple containing the value, key, and a boolean indicating if the field is complex.
        """
        # Not used directly by Pydantic internals in the way we want for full dict loading
        # standard implementation requires implementing __call__
        return None, "", False  # pragma: no cover

    def __call__(self) -> Dict[str, Any]:
        """Load the YAML file and return as a dict.

        Returns:
            Dict[str, Any]: The configuration dictionary loaded from YAML.

        Raises:
            ValueError: If the configuration file is invalid.
        """
        config_path = os.getenv("SEARCH_CONFIG_PATH", "search_config.yaml")
        path = Path(config_path)

        if not path.exists():
            # If default file missing, return empty (use defaults)
            if config_path == "search_config.yaml":
                return {}
            # If explicit path missing, warn but maybe return empty or error?
            # Previous logic returned empty with defaults.
            logger.warning(f"Config file not found at {config_path}")
            return {}

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            # We want to propagate format errors as ValueError for compatibility
            raise ValueError(f"Invalid configuration file: {e}") from e


class Settings(BaseSettings):
    """Root configuration for the search application.

    Aggregates configurations for embeddings, re-ranking, scout, and database.
    Supports loading from YAML files and environment variables.
    Env vars take precedence: e.g. APP__DATABASE_URI overrides yaml.

    Attributes:
        embedding: Configuration for the embedding model.
        reranker: Configuration for the re-ranking model.
        scout: Configuration for the scout.
        database_uri: The URI for the LanceDB database.
        env: The current environment (e.g., 'development', 'production').
    """

    model_config = SettingsConfigDict(
        env_prefix="APP__",
        env_nested_delimiter="__",
        frozen=True,
        extra="ignore",  # Ignore extra fields in yaml/env
    )

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    scout: ScoutConfig = Field(default_factory=ScoutConfig)
    database_uri: str = Field(default="/tmp/lancedb")
    env: str = Field(default="development")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Define the priority of settings sources.

        1. Init arguments (highest)
        2. Environment variables
        3. YAML file
        4. Defaults (lowest)

        Args:
            settings_cls: The settings class.
            init_settings: Settings from init arguments.
            env_settings: Settings from environment variables.
            dotenv_settings: Settings from dotenv files.
            file_secret_settings: Settings from file secrets.

        Returns:
            Tuple[PydanticBaseSettingsSource, ...]: The ordered settings sources.
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_config(config_path: str | None = None) -> Settings:
    """Backward-compatible helper to load settings.

    If config_path is provided, it sets the env var momentarily or constructs Settings directly.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        Settings: The loaded configuration object.
    """
    if config_path:
        os.environ["SEARCH_CONFIG_PATH"] = config_path

    # Reload settings
    return Settings()
