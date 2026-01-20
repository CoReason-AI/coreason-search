# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import re
from typing import Dict, List

# Map common PubMed tags to Tantivy fields
FIELD_MAPPING: Dict[str, str] = {
    "ti": "title",
    "title": "title",
    "ab": "abstract",
    "abstract": "abstract",
    "tiab": "title_abstract",  # Special handling might be needed
    "mesh": "mesh_terms",
    "mh": "mesh_terms",
}


def _map_tags_to_fields(tags: List[str]) -> List[str]:
    """Helper to map PubMed tags to Tantivy fields."""
    mapped_fields = []
    for t in tags:
        # Check mapping or use raw if not found
        field = FIELD_MAPPING.get(t, t)

        # Special case for TiAb -> expand to Title OR Abstract
        if field == "title_abstract":
            mapped_fields.append("title")
            mapped_fields.append("abstract")
        else:
            mapped_fields.append(field)
    return mapped_fields


def parse_pubmed_query(query: str) -> str:
    """Translate a PubMed-style Boolean query into a Tantivy-compatible query string.

    Examples:
        "Aspirin"[Title] -> title:"Aspirin"
        Aspirin[Title] -> title:Aspirin
        "Heart Attack"[Ti] -> title:"Heart Attack"
        ("A"[Ti] OR "B"[Ab]) -> (title:"A" OR abstract:"B")

    Args:
        query: The raw PubMed-style query string.

    Returns:
        str: The Tantivy-compatible query string.
    """
    if not query:
        return ""  # pragma: no cover

    # Regex to capture term[tag]
    # Group 1: Quoted term (e.g. "Aspirin")
    # Group 2: Quote char
    # Group 3: Unquoted term (e.g. Aspirin)
    # Group 4: Tag (e.g. Title)

    # Pattern explanation:
    # (["'])(.*?)\2  -> Matches quoted string: "term" or 'term'
    # |              -> OR
    # ([^\s()\[\]]+) -> Matches unquoted term: term (no spaces, parens, brackets)
    # )              -> End term group
    # \s*            -> Optional whitespace
    # \[             -> Literal [
    # (.*?)          -> Capture tag
    # \]             -> Literal ]

    pattern = re.compile(r'(?:(["\'])(.*?)\1|([^\s()\[\]]+))\s*\[(.*?)\]')

    def replace_match(match: re.Match[str]) -> str:
        # Check which group matched
        # quote_char = match.group(1)
        quoted_content = match.group(2)
        unquoted_term = match.group(3)
        tag_raw = match.group(4)

        # Determine the term
        if unquoted_term is not None:
            term = unquoted_term
        else:
            # Reconstruct quoted term (Tantivy usually handles standard double quotes)
            # Use double quotes for consistency
            term = f'"{quoted_content}"'

        # Parse tags (handle slashes e.g. Title/Abstract)
        tags = [t.strip().lower() for t in tag_raw.split("/")]

        mapped_fields = _map_tags_to_fields(tags)

        # Construct result
        # If multiple fields, wrap in parens with OR
        if len(mapped_fields) > 1:
            clauses = [f"{f}:{term}" for f in mapped_fields]
            return f"({' OR '.join(clauses)})"
        elif len(mapped_fields) == 1:
            return f"{mapped_fields[0]}:{term}"
        else:  # pragma: no cover
            # Fallback (no tag matched?) - return term as is (search all fields)
            return term

    # Apply replacement
    result = pattern.sub(replace_match, query)

    return result
