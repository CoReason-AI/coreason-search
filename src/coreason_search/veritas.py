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
from typing import Any, Dict, Optional

from coreason_search.utils.logger import logger


class VeritasProtocol(ABC):
    """
    Protocol for the Coreason Veritas Audit System.
    Ensures research-grade reproducibility and auditability.
    """

    @abstractmethod
    def log_audit(self, event: str, data: Dict[str, Any]) -> None:
        """
        Log an audit event.

        Args:
            event: The event name (e.g., "SYSTEMATIC_SEARCH_START").
            data: The structured data payload.
        """
        pass  # pragma: no cover


class MockVeritasClient(VeritasProtocol):
    """
    Mock implementation of Veritas Client.
    Logs structured JSON to the standard logger.
    """

    def log_audit(self, event: str, data: Dict[str, Any]) -> None:
        """
        Log the audit event to the application logger with a specific structure.
        In a real implementation, this would send data to the Veritas service.
        """
        audit_payload = {
            "component": "coreason-search",
            "event": event,
            "data": data,
        }
        logger.info(f"VERITAS_AUDIT: {audit_payload}")


_veritas_instance: Optional[VeritasProtocol] = None


def get_veritas_client() -> VeritasProtocol:
    """Singleton factory for Veritas Client."""
    global _veritas_instance
    if _veritas_instance is None:
        _veritas_instance = MockVeritasClient()
    return _veritas_instance


def reset_veritas_client() -> None:
    """Reset singleton (for testing)."""
    global _veritas_instance
    _veritas_instance = None
