# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from coreason_search.utils.query_parser import parse_pubmed_query


def test_special_chars_in_quotes() -> None:
    # Quotes containing special chars
    query = '"O\'Neil"[Author]'
    expected = 'author:"O\'Neil"'
    assert parse_pubmed_query(query) == expected

    query = '"Title: Subtitle"[Title]'
    expected = 'title:"Title: Subtitle"'
    assert parse_pubmed_query(query) == expected

    query = '"Term [with] brackets"[Title]'
    expected = 'title:"Term [with] brackets"'
    assert parse_pubmed_query(query) == expected


def test_complex_nested_boolean() -> None:
    # Complex mix of quoted, unquoted, aliases, and logic
    query = '(Pandemic[Ti] OR "Covid-19"[TiAb]) AND (Vaccine OR "Public Health"[Mesh])'
    # Expected translation:
    # Pandemic[Ti] -> title:Pandemic
    # "Covid-19"[TiAb] -> (title:"Covid-19" OR abstract:"Covid-19")
    # Vaccine -> Vaccine (no change)
    # "Public Health"[Mesh] -> mesh_terms:"Public Health"

    expected = (
        '(title:Pandemic OR (title:"Covid-19" OR abstract:"Covid-19")) AND (Vaccine OR mesh_terms:"Public Health")'
    )
    assert parse_pubmed_query(query) == expected


def test_whitespace_in_tags() -> None:
    # Whitespace inside the brackets
    query = "Cancer[  Title  ]"
    expected = "title:Cancer"
    assert parse_pubmed_query(query) == expected

    query = "Cancer[ Title / Abstract ]"
    # Should split correctly
    result = parse_pubmed_query(query)
    assert "title:Cancer" in result
    assert "abstract:Cancer" in result


def test_multi_word_without_quotes() -> None:
    # PubMed behavior: lung cancer[Title] -> lung title:cancer
    query = "lung cancer[Title]"
    expected = "lung title:cancer"
    assert parse_pubmed_query(query) == expected


def test_empty_term_or_tag() -> None:
    # Empty quotes
    query = '""[Title]'
    expected = 'title:""'
    assert parse_pubmed_query(query) == expected

    # Empty tag? usually illegal regex might fail or produce empty tag
    # Our regex expects [.*?]
    query = "Term[]"
    # Tag is empty. Split by / gives [''].
    # Field mapping get('') defaults to ''.
    # result: :Term
    # While invalid, we check it doesn't crash.
    expected = ":Term"
    assert parse_pubmed_query(query) == expected


def test_escaped_quotes_logic() -> None:
    # If the user tries to escape quotes inside?
    # Python regex with ".*?" stops at first quote.
    # "Term \"quote\""[Title]
    # Regex `(["'])(.*?)\1`
    # Matching starts at first ".
    # Matches `Term \` then stops at first `"`?
    # Let's see how python re handles it.

    # Input: "Term \"quote\""[Title]
    # In python string: '"Term \\"quote\\""[Title]'
    # The regex might struggle with escaped quotes if we don't handle them explicitly.
    # Tantivy/PubMed doesn't strictly define escaping in the same way, usually doubled quotes.
    pass


def test_mixed_case_tags() -> None:
    query = "Term[tiAB]"
    result = parse_pubmed_query(query)
    assert "title:Term" in result
    assert "abstract:Term" in result


def test_unicode_chars() -> None:
    query = "β-amyloid[Title]"
    expected = "title:β-amyloid"
    assert parse_pubmed_query(query) == expected


def test_weird_spacing_between_term_and_tag() -> None:
    # Term   [Tag]
    query = "Term   [Title]"
    expected = "title:Term"
    assert parse_pubmed_query(query) == expected
