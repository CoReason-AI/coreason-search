# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Any, Dict, Union


def extract_query_text(query: Union[str, Dict[str, Any]]) -> str:
    """Extract a string representation of the query for semantic search or logging.

    If query is a Dict, tries to use 'text' key, otherwise joins all values.

    Args:
        query: The raw query from SearchRequest.

    Returns:
        str: Normalized query string.
    """
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        if "text" in query:
            return str(query["text"])
        # Join values as fallback
        return " ".join(str(v) for v in query.values())
    return str(query)
