# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from coreason_search.schemas import Hit
from coreason_search.scout import MockScout


def test_mock_scout_distillation() -> None:
    """Test that MockScout correctly 'distills' text."""
    scout = MockScout()

    original_text = "This is a very long sentence that needs to be distilled."
    hit = Hit(
        doc_id="1",
        content="preview",
        original_text=original_text,
        distilled_text="",  # Initially empty or same as original
        score=1.0,
        source_strategy="test",
        metadata={},
    )

    hits = [hit]
    result_hits = scout.distill(query="test", hits=hits)

    assert len(result_hits) == 1
    distilled = result_hits[0].distilled_text

    # Verify it changed (mock logic halves the string)
    assert len(distilled) < len(original_text)
    assert distilled.endswith("...")

    # Verify original text is untouched
    assert result_hits[0].original_text == original_text


def test_mock_scout_empty_hits() -> None:
    """Test MockScout with empty list."""
    scout = MockScout()
    result_hits = scout.distill(query="test", hits=[])
    assert result_hits == []
