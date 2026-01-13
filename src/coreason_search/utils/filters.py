# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Any, Dict


def matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if the metadata matches the MongoDB-style filters.

    Args:
        metadata: The document metadata dictionary.
        filters: The filter dictionary (e.g. {"year": {"$gt": 2020}, "category": "paper"}).

    Returns:
        bool: True if it matches all filters, False otherwise.
    """
    for key, condition in filters.items():
        # Get the value from metadata, default to None if missing
        value = metadata.get(key)

        if isinstance(condition, dict):
            # Handle operators
            if not _check_condition_operators(value, condition):
                return False
        else:
            # Direct equality
            if value != condition:
                return False

    return True


def _check_condition_operators(value: Any, condition: Dict[str, Any]) -> bool:
    """Helper to check operators for a single field."""
    for op, target in condition.items():
        if op == "$eq":
            if value != target:
                return False
        elif op == "$ne":
            if value == target:
                return False
        elif op == "$gt":
            if value is None or not (value > target):
                return False
        elif op == "$gte":
            if value is None or not (value >= target):
                return False
        elif op == "$lt":
            if value is None or not (value < target):
                return False
        elif op == "$lte":
            if value is None or not (value <= target):
                return False
        elif op == "$in":
            if not isinstance(target, list):
                # Should be a list, if not, maybe treat as single value?
                # But strict MongoDB implies list.
                if value != target:
                    return False
            else:
                if value not in target:
                    return False
        elif op == "$nin":
            if isinstance(target, list) and value in target:
                return False
        else:
            # Unknown operator, assume direct match if it's not starting with $?
            # Or fail? Let's ignore or treat as False to be safe.
            # But nested structure in filters usually implies operators.
            pass

    return True
