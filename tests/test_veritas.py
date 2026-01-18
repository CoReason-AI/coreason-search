# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Generator
from unittest.mock import patch

import pytest

from coreason_search.veritas import MockVeritasClient, get_veritas_client, reset_veritas_client


class TestVeritas:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self) -> Generator[None, None, None]:
        reset_veritas_client()
        yield
        reset_veritas_client()

    def test_singleton(self) -> None:
        client1 = get_veritas_client()
        client2 = get_veritas_client()
        assert client1 is client2
        assert isinstance(client1, MockVeritasClient)

    def test_log_audit(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that log_audit logs correct JSON structure to logger."""
        client = get_veritas_client()

        event = "TEST_EVENT"
        data = {"key": "value"}

        # We need to use proper logger capture for loguru
        # Using pytest-caplog with loguru requires some setup or just checking stderr
        # But assuming pytest-caplog integration is set up via proper fixtures or we check output

        # For this test, we can mock the logger inside veritas or rely on caplog
        # if propogation is enabled. Loguru usually writes to stderr.
        # Let's mock logger.info

        with patch("coreason_search.veritas.logger") as mock_logger:
            client.log_audit(event, data)
            mock_logger.info.assert_called_once()
            args = mock_logger.info.call_args[0][0]
            assert "TEST_EVENT" in args
            assert "key" in args
            assert "value" in args
            assert "coreason-search" in args
