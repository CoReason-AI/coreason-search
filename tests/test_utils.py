# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from pathlib import Path

from coreason_search.utils.filters import check_single_op, matches_filters
from coreason_search.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly."""
    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_matches_filters_simple() -> None:
    """Test simple equality filters."""
    meta = {"year": 2024, "category": "A"}
    assert matches_filters(meta, {"year": 2024})
    assert matches_filters(meta, {"category": "A"})
    assert not matches_filters(meta, {"year": 2023})


def test_matches_filters_operators() -> None:
    """Test standard operators."""
    meta = {"year": 2024, "score": 10.5, "tags": ["a", "b"], "empty": None}

    # $eq
    assert matches_filters(meta, {"year": {"$eq": 2024}})
    assert not matches_filters(meta, {"year": {"$eq": 2023}})

    # $gt / $gte
    assert matches_filters(meta, {"year": {"$gt": 2023}})
    assert matches_filters(meta, {"year": {"$gte": 2024}})
    assert not matches_filters(meta, {"year": {"$gt": 2024}})

    # $lt / $lte
    assert matches_filters(meta, {"score": {"$lt": 11}})
    assert matches_filters(meta, {"score": {"$lte": 10.5}})

    # $ne
    assert matches_filters(meta, {"year": {"$ne": 2020}})

    # $in / $nin
    assert matches_filters(meta, {"year": {"$in": [2023, 2024]}})
    assert not matches_filters(meta, {"year": {"$in": [2020, 2021]}})
    assert matches_filters(meta, {"year": {"$nin": [2020, 2021]}})


def test_matches_filters_nested() -> None:
    """Test dot notation support."""
    meta = {"author": {"name": "Smith", "age": 40}, "info": {"year": 2024}}

    assert matches_filters(meta, {"author.name": "Smith"})
    assert matches_filters(meta, {"author.age": {"$gt": 30}})
    assert matches_filters(meta, {"info.year": 2024})

    # Missing nested key
    assert not matches_filters(meta, {"author.gender": "M"})
    assert matches_filters(meta, {"author.gender": {"$ne": "M"}})  # Missing matches != M? Yes, None != M


def test_matches_filters_logical() -> None:
    """Test logical operators ($or, $and, $not)."""
    meta = {"year": 2024, "category": "A"}

    # $or
    assert matches_filters(meta, {"$or": [{"year": 2024}, {"year": 2025}]})
    assert matches_filters(meta, {"$or": [{"year": 2020}, {"category": "A"}]})
    assert not matches_filters(meta, {"$or": [{"year": 2020}, {"category": "B"}]})

    # $and (redundant but explicit)
    assert matches_filters(meta, {"$and": [{"year": 2024}, {"category": "A"}]})
    assert not matches_filters(meta, {"$and": [{"year": 2024}, {"category": "B"}]})

    # $not
    assert matches_filters(meta, {"$not": {"year": 2025}})
    assert not matches_filters(meta, {"$not": {"year": 2024}})


def test_matches_filters_list_awareness() -> None:
    """Test implicit list matching (Mongo style)."""
    meta = {"tags": ["science", "fiction"]}

    # Equality matches item in list
    assert matches_filters(meta, {"tags": "science"})
    assert matches_filters(meta, {"tags": "fiction"})
    assert not matches_filters(meta, {"tags": "horror"})

    # But exact list match works too?
    assert matches_filters(meta, {"tags": ["science", "fiction"]})


def test_matches_filters_type_safety() -> None:
    """Test robustness against type errors."""
    meta = {"year": "2024"}  # String year

    # Compare with int
    # "2024" > 2023 (TypeError in Python 3) -> Should return False (safe)
    assert not matches_filters(meta, {"year": {"$gt": 2023}})

    # Check that it didn't crash
    assert True


def test_malformed_filters() -> None:
    """Test malformed filters."""
    meta = {"a": 1}
    # $or not list
    assert not matches_filters(meta, {"$or": "not-a-list"})
    # $and not list
    assert not matches_filters(meta, {"$and": "not-a-list"})
    # $in not list
    assert not matches_filters(meta, {"a": {"$in": "not-a-list"}})


def test_coverage_edge_cases() -> None:
    """Explicit tests for edge cases to ensure 100% coverage."""
    # 1. Nested path where intermediate key is not a dict
    meta = {"a": 1}
    # Path "a.b". "a" is 1 (int), not dict. Should return None.
    assert not matches_filters(meta, {"a.b": 1})

    # 2. $in with scalar target (fallback to equality)
    # meta={"a": "foo"}, filter={"a": {"$in": "foo"}}
    # "foo" is str (not list/tuple). checks equality. "foo"=="foo".
    assert matches_filters({"a": "foo"}, {"a": {"$in": "foo"}})
    # Mismatch case
    assert not matches_filters({"a": "bar"}, {"a": {"$in": "foo"}})

    # 3. $nin with value matching target
    # meta={"a": 1}, filter={"a": {"$nin": [1, 2]}}
    # 1 in [1, 2]. returns False.
    assert not matches_filters({"a": 1}, {"a": {"$nin": [1, 2]}})

    # 4. Explicit operator false branches just to be sure
    assert not matches_filters({"a": 1}, {"a": {"$gt": 2}})
    assert not matches_filters({"a": 1}, {"a": {"$gte": 2}})
    assert not matches_filters({"a": 1}, {"a": {"$lt": 0}})
    assert not matches_filters({"a": 1}, {"a": {"$lte": 0}})
    assert not matches_filters({"a": 1}, {"a": {"$eq": 2}})


def test_check_single_op_direct() -> None:
    """Direct tests for check_single_op to ensure coverage."""
    # $in with list
    assert check_single_op("$in", 1, [1, 2])
    assert not check_single_op("$in", 3, [1, 2])

    # $nin with list
    assert not check_single_op("$nin", 1, [1, 2])
    assert check_single_op("$nin", 3, [1, 2])

    # $in fallback
    assert check_single_op("$in", 1, 1)
    assert not check_single_op("$in", 1, 2)

    # $nin fallback
    assert not check_single_op("$nin", 1, 1)
    assert check_single_op("$nin", 1, 2)


def test_check_single_op_unknown() -> None:
    """Test unknown operator falls through to True."""
    assert check_single_op("$unknown", 1, 1)
