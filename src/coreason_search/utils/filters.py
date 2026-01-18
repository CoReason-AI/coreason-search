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
    Supports dot notation for nested fields and logical operators ($or, $and, $not).

    Args:
        metadata: The document metadata dictionary.
        filters: The filter dictionary.

    Returns:
        bool: True if it matches all filters, False otherwise.
    """
    # 1. Handle Logical Operators ($or, $and, $not)
    if "$or" in filters:
        conditions = filters["$or"]
        if isinstance(conditions, list):
            if not any(matches_filters(metadata, cond) for cond in conditions):
                return False
        else:
            return False

    if "$and" in filters:
        conditions = filters["$and"]
        if isinstance(conditions, list):
            if not all(matches_filters(metadata, cond) for cond in conditions):
                return False
        else:
            return False

    if "$not" in filters:
        condition = filters["$not"]
        if matches_filters(metadata, condition):
            return False

    # 2. Handle Field Constraints
    for key, condition in filters.items():
        if key.startswith("$"):
            continue

        value = _get_value_by_path(metadata, key)

        if isinstance(condition, dict):
            if not _check_condition_operators(value, condition):
                return False
        else:
            # Direct equality with implicit list match
            if isinstance(value, list) and not isinstance(condition, list):
                if condition not in value:
                    return False
            else:
                if value != condition:
                    return False

    return True


def _get_value_by_path(data: Any, path: str) -> Any:
    """Retrieve value from nested dict using dot notation."""
    keys = path.split(".")
    curr = data
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k)
        else:
            return None
    return curr


def _check_condition_operators(value: Any, condition: Dict[str, Any]) -> bool:
    """Helper to check operators for a single field with type safety."""
    for op, target in condition.items():
        if not check_single_op(op, value, target):
            return False
    return True


def check_single_op(op: str, value: Any, target: Any) -> bool:
    """Check a single operator condition."""
    try:
        if op == "$eq":
            return bool(value == target)
        if op == "$ne":
            return bool(value != target)
        if op == "$gt":
            return value is not None and bool(value > target)
        if op == "$gte":
            return value is not None and bool(value >= target)
        if op == "$lt":
            return value is not None and bool(value < target)
        if op == "$lte":
            return value is not None and bool(value <= target)
        if op == "$in":
            if isinstance(target, (list, tuple)):
                return bool(value in target)  # pragma: no cover
            return bool(value == target)
        if op == "$nin":
            if isinstance(target, (list, tuple)):
                return bool(value not in target)  # pragma: no cover
            return bool(value != target)
    except TypeError:
        # Type mismatch (e.g. comparing str > int) returns False
        return False  # pragma: no cover

    # Unknown operator treated as True
    return True
