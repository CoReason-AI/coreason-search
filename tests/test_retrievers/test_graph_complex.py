# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import cast

from coreason_search.graph_client import GraphNode, MockGraphClient, reset_graph_client
from coreason_search.retrievers.graph import GraphRetriever
from coreason_search.schemas import RetrieverType, SearchRequest


class TestGraphRetrieverComplex:
    def setup_method(self) -> None:
        """
        Setup a complex graph topology for testing.
        Topology:
        - Protein A (Query Match)
          -> Paper 1 (Valid) -> AE 1, AE 2
          -> Paper 2 (Invalid - No AE) -> Protein B
          -> Paper 3 (Valid - Shared AE) -> AE 2
          -> Unknown Node (Invalid Label)
        - Protein B (Not Query Match)
          -> Paper 4 (Valid but unreachable) -> AE 3
        """
        reset_graph_client()
        self.retriever = GraphRetriever()
        self.client = cast(MockGraphClient, self.retriever.client)
        assert isinstance(self.client, MockGraphClient)

        # clear default mock data
        self.client.nodes = {}
        self.client.edges = []

        # 1. Create Nodes
        nodes = [
            # Entry points
            GraphNode(node_id="p_a", label="Protein", name="Protein A", properties={"desc": "Target A"}),
            GraphNode(node_id="p_b", label="Protein", name="Protein B", properties={"desc": "Target B"}),
            # Papers
            GraphNode(
                node_id="paper_1",
                label="Paper",
                name="Paper 1",
                properties={"content": "Content 1", "year": 2021},
            ),
            GraphNode(
                node_id="paper_2",
                label="Paper",
                name="Paper 2",
                properties={"content": "Content 2", "year": 2022},
            ),
            GraphNode(
                node_id="paper_3",
                label="Paper",
                name="Paper 3",
                properties={"content": "Content 3", "year": 2023},
            ),
            GraphNode(
                node_id="paper_4",
                label="Paper",
                name="Paper 4",
                properties={"content": "Content 4", "year": 2024},
            ),
            # Adverse Events
            GraphNode(node_id="ae_1", label="AdverseEvent", name="Nausea", properties={}),
            GraphNode(node_id="ae_2", label="AdverseEvent", name="Headache", properties={}),
            GraphNode(node_id="ae_3", label="AdverseEvent", name="Dizziness", properties={}),
            # Distractor
            GraphNode(node_id="unk_1", label="Unknown", name="Mystery", properties={}),
        ]
        for n in nodes:
            self.client.nodes[n.node_id] = n

        # 2. Create Edges
        edges = [
            # Protein A connections
            ("p_a", "paper_1"),
            ("p_a", "paper_2"),
            ("p_a", "paper_3"),
            ("p_a", "unk_1"),
            # Paper 1 -> AE 1, AE 2
            ("paper_1", "ae_1"),
            ("paper_1", "ae_2"),
            # Paper 2 -> Protein B (Not an AE)
            ("paper_2", "p_b"),
            # Paper 3 -> AE 2 (Shared)
            ("paper_3", "ae_2"),
            # Protein B -> Paper 4
            ("p_b", "paper_4"),
            # Paper 4 -> AE 3
            ("paper_4", "ae_3"),
        ]
        self.client.edges = [{"source": s, "target": t} for s, t in edges]

    def test_complex_traversal_and_enrichment(self) -> None:
        """
        Verify:
        - Only papers reachable from Query Node are returned.
        - Only papers connected to an Adverse Event are returned.
        - Metadata is correctly enriched with sorted AE lists.
        """
        request = SearchRequest(
            query="Protein A",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )
        hits = self.retriever.retrieve(request)

        # Sort hits by doc_id to ensure deterministic assertions
        hits.sort(key=lambda x: x.doc_id)

        # Expect Paper 1 and Paper 3.
        # Paper 2 is rejected (no AE).
        # Paper 4 is rejected (not reachable from Protein A).
        assert len(hits) == 2

        # Check Paper 1
        h1 = hits[0]
        assert h1.doc_id == "paper_1"
        assert h1.content == "Content 1"
        # Check original metadata preserved
        assert h1.metadata["year"] == 2021
        # Check enriched metadata
        assert h1.metadata["connected_adverse_events"] == ["Headache", "Nausea"]  # Alphabetical

        # Check Paper 3
        h2 = hits[1]
        assert h2.doc_id == "paper_3"
        assert h2.metadata["connected_adverse_events"] == ["Headache"]

    def test_cycle_robustness(self) -> None:
        """
        Test that cycles in the graph do not cause infinite loops.
        Although logic is 2-hop fixed depth, verify it handles self-references.
        Add Edge: Paper 1 -> Protein A (Cycle)
        """
        self.client.edges.append({"source": "paper_1", "target": "p_a"})

        request = SearchRequest(
            query="Protein A",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )
        hits = self.retriever.retrieve(request)
        # Should still return results without error
        assert len(hits) >= 1
        ids = [h.doc_id for h in hits]
        assert "paper_1" in ids

    def test_empty_metadata_handling(self) -> None:
        """
        Test that a paper with empty properties is handled correctly.
        """
        # Create a paper with no properties
        self.client.nodes["paper_empty"] = GraphNode(
            node_id="paper_empty",
            label="Paper",
            name="Empty Paper",
            properties={},
        )
        self.client.edges.append({"source": "p_a", "target": "paper_empty"})
        self.client.edges.append({"source": "paper_empty", "target": "ae_1"})

        request = SearchRequest(
            query="Protein A",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )
        hits = self.retriever.retrieve(request)
        ids = [h.doc_id for h in hits]
        assert "paper_empty" in ids
        hit = next(h for h in hits if h.doc_id == "paper_empty")
        assert hit.content == ""  # Content defaults to empty string if missing
        assert hit.metadata["connected_adverse_events"] == ["Nausea"]

    def test_multiple_start_nodes(self) -> None:
        """
        Test query matching multiple start nodes.
        Query: "Protein" (Matches "Protein A" and "Protein B")
        """
        request = SearchRequest(
            query="Protein",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )
        hits = self.retriever.retrieve(request)
        hits_ids = {h.doc_id for h in hits}

        # Protein A -> Paper 1, Paper 3
        # Protein B -> Paper 4
        assert "paper_1" in hits_ids
        assert "paper_3" in hits_ids
        assert "paper_4" in hits_ids
        assert "paper_2" not in hits_ids

    def test_mock_client_hop_depth_ignored(self) -> None:
        """
        Test that MockGraphClient handles hop_depth parameter safely.
        """
        # Call get_neighbors with depth != 1
        neighbors = self.client.get_neighbors("p_a", hop_depth=2)
        # Mock currently ignores depth > 1 but should still return direct neighbors
        # It just does the same thing.
        assert len(neighbors) > 0
