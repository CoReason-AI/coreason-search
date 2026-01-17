# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import json
from typing import Any, Dict

from coreason_search.schemas import Hit


class LanceMapper:
    """Helper to map LanceDB results to Hit objects."""

    @staticmethod
    def map_hit(item: Dict[str, Any], source_strategy: str, score: float = 0.0) -> Hit:
        """
        Map a single LanceDB result item to a Hit object.

        Args:
            item: The dictionary returned by LanceDB.
            source_strategy: The retriever strategy name.
            score: The relevance score.

        Returns:
            Hit: The populated Hit object.
        """
        doc_id = item["doc_id"]
        content = item["content"]
        metadata_str = item["metadata"]

        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:  # pragma: no cover
            metadata = {}

        return Hit(
            doc_id=doc_id,
            content=content,
            original_text=content,
            distilled_text="",  # Populated later
            score=score,
            source_strategy=source_strategy,
            metadata=metadata,
        )
