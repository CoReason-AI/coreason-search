# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from coreason_search.utils.common import extract_query_text


def test_extract_query_text_str() -> None:
    assert extract_query_text("foo") == "foo"


def test_extract_query_text_dict_with_text() -> None:
    assert extract_query_text({"text": "foo", "other": "bar"}) == "foo"


def test_extract_query_text_dict_without_text() -> None:
    # This hits the fallback line
    # Values order in dict is insertion order in recent Python, but let's be safe
    q = {"a": "foo", "b": "bar"}
    res = extract_query_text(q)
    assert "foo" in res
    assert "bar" in res


def test_extract_query_text_other() -> None:
    assert extract_query_text(123) == "123"  # type: ignore[arg-type]
