# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict

from coreason_search.utils.logger import logger


class VeritasProtocol(ABC):
    """Protocol for the Coreason Veritas Audit System.

    Ensures research-grade reproducibility and auditability.
    """

    @abstractmethod
    def log_audit(self, event: str, data: Dict[str, Any]) -> None:
        """Log an audit event.

        Args:
            event: The event name (e.g., "SYSTEMATIC_SEARCH_START").
            data: The structured data payload.
        """
        pass  # pragma: no cover


class MockVeritasClient(VeritasProtocol):
    """Mock implementation of Veritas Client.

    Logs structured JSON to the standard logger.
    """

    def log_audit(self, event: str, data: Dict[str, Any]) -> None:
        """Log the audit event to the application logger with a specific structure.

        In a real implementation, this would send data to the Veritas service.

        Args:
            event: The event name.
            data: The audit data.
        """
        audit_payload = {
            "component": "coreason-search",
            "event": event,
            "data": data,
        }
        logger.info(f"VERITAS_AUDIT: {audit_payload}")


@lru_cache(maxsize=32)
def get_veritas_client() -> VeritasProtocol:
    """Singleton factory for Veritas Client.

    Returns:
        VeritasProtocol: An instance of the veritas client.
    """
    return MockVeritasClient()


def reset_veritas_client() -> None:
    """Reset singleton (for testing)."""
    get_veritas_client.cache_clear()
