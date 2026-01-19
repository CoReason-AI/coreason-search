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
from typing import Optional

import yaml

from coreason_search.schemas import AppConfig
from coreason_search.utils.logger import logger


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from a YAML file.
    If config_path is not provided, tries to read SEARCH_CONFIG_PATH env var,
    defaulting to 'search_config.yaml'.
    If the file does not exist, returns default AppConfig.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        AppConfig: The populated configuration object.
    """
    if not config_path:
        config_path = os.getenv("SEARCH_CONFIG_PATH", "search_config.yaml")

    if not os.path.exists(config_path):
        logger.info(f"Config file not found at {config_path}, using defaults.")
        return AppConfig()

    logger.info(f"Loading config from {config_path}")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Validate with Pydantic
        config = AppConfig(**data)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise ValueError(f"Invalid configuration file: {e}") from e
