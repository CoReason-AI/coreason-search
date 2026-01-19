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
    model_config = ConfigDict(frozen=True)

    provider: Literal["auto", "hf", "mock"] = Field(
        default="auto", description="Embedder provider: 'auto', 'hf', 'mock'"
    )
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    context_length: int = Field(default=32768, gt=0)
    batch_size: int = Field(default=1, gt=0)


class RerankerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"


class ScoutConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a YAML file
    at the project's root or specified location.
    """

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        # Not used directly by Pydantic internals in the way we want for full dict loading
        # standard implementation requires implementing __call__
        return None, "", False  # pragma: no cover

    def __call__(self) -> Dict[str, Any]:
        """
        Load the YAML file and return as a dict.
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
    """
    Root configuration for the search application.
    Supports loading from 'search_config.yaml' and Environment Variables.
    Env vars take precedence: e.g. APP__DATABASE_URI overrides yaml.
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
        """
        Define the priority of settings sources.
        1. Init arguments (highest)
        2. Environment variables
        3. YAML file
        4. Defaults (lowest)
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_config(config_path: str | None = None) -> Settings:
    """
    Backward-compatible helper to load settings.
    If config_path is provided, it sets the env var momentarily or constructs Settings directly.
    Since Pydantic Settings reads the file from the class definition (via source),
    we need to trick it if we want to change the file path dynamically for tests.
    """
    if config_path:
        os.environ["SEARCH_CONFIG_PATH"] = config_path

    # Reload settings
    return Settings()
