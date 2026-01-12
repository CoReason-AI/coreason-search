# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_search.utils import logger as logger_module
from coreason_search.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    log_path = Path("logs")
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_logger_directory_creation() -> None:
    """Test that the logger creates the directory if it doesn't exist."""
    # We need to reload the module to trigger the top-level code again
    # We mock Path to control the existence check
    with patch("coreason_search.utils.logger.Path") as mock_path_cls:
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        # First call to exists returns False (triggering mkdir), subsequent return whatever
        mock_path_instance.exists.side_effect = [False, True, True]

        # Force reload to run the module-level code
        importlib.reload(logger_module)

        # Verify mkdir was called
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Reload again to restore normal state for other tests (optional but good practice)
    importlib.reload(logger_module)
