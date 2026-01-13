# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from coreason_search.graph_client import MockGraphClient, reset_graph_client


class TestGraphClient:
    def setup_method(self) -> None:
        reset_graph_client()

    def test_search_nodes(self) -> None:
        client = MockGraphClient()
        nodes = client.search_nodes("Protein X")
        # In mock data: "Protein X", "Study on Protein X", "Protein X is safe" (desc in content)
        # Search is substring on NAME.
        # "Protein X" (name="Protein X") -> Match
        # "paper_a" (name="Study on Protein X") -> Match
        # "paper_b" (name="Another Study") -> No Match
        # So we expect 2 matches if substring logic is: `if query_lower in node.name.lower():`
        assert len(nodes) >= 1
        names = sorted([n.name for n in nodes])
        assert "Protein X" in names
        assert "Study on Protein X" in names

    def test_search_no_match(self) -> None:
        client = MockGraphClient()
        nodes = client.search_nodes("Nonexistent")
        assert len(nodes) == 0

    def test_get_neighbors(self) -> None:
        client = MockGraphClient()
        # Protein X -> Paper A, Paper B
        neighbors = client.get_neighbors("protein_x")
        assert len(neighbors) == 2
        # Mock client uses "source -> target" for this direction
        names = sorted([n.name for n in neighbors])
        assert names == ["Another Study", "Study on Protein X"]

    def test_get_neighbors_incoming(self) -> None:
        # Paper A -> Liver Failure
        # Ask neighbors of Liver Failure (incoming edge check)
        # Mock implementation checks both source==id and target==id
        client = MockGraphClient()
        neighbors = client.get_neighbors("liver_failure")
        assert len(neighbors) == 1
        assert neighbors[0].name == "Study on Protein X"

    def test_get_neighbors_depth_ignored(self) -> None:
        """Test that depth parameter is accepted (even if ignored by mock logic)."""
        client = MockGraphClient()
        neighbors = client.get_neighbors("protein_x", hop_depth=2)
        assert len(neighbors) == 2
