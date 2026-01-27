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
from coreason_search.retrievers.graph import GraphRetriever
from coreason_search.schemas import RetrieverType, SearchRequest


class TestGraphRetriever:
    def setup_method(self) -> None:
        reset_graph_client()

    def test_retrieve_found(self) -> None:
        """Test retrieving papers via graph traversal."""
        retriever = GraphRetriever()
        request = SearchRequest(
            query="Protein X",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )

        hits = retriever.retrieve(request)
        # Mock Graph: Protein X -> Paper A, Paper B.
        # Paper A -> Liver Failure (AdverseEvent).
        # Paper B -> (No AdverseEvent).
        # New 2-hop logic filters Paper B.
        assert len(hits) == 1
        assert hits[0].doc_id == "paper_a"

        # Verify hit structure
        assert hits[0].source_strategy == "graph_neighbor"
        assert hits[0].score == 1.0
        assert hits[0].content is not None
        assert "This paper discusses Protein X and liver failure." in hits[0].content

    def test_retrieve_no_node(self) -> None:
        """Test when initial node search fails."""
        retriever = GraphRetriever()
        request = SearchRequest(query="Unknown Thing", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)
        assert len(hits) == 0

    def test_retrieve_dict_query(self) -> None:
        """Test dict query handling (fallback to string)."""
        retriever = GraphRetriever()
        # Should convert {"entity": "Protein X"} -> "Protein X" (via values join)
        request = SearchRequest(
            query={"entity": "Protein X"},
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )
        hits = retriever.retrieve(request)
        # Same as test_retrieve_found, expects 1 hit due to 2-hop filtering
        assert len(hits) == 1
        assert hits[0].doc_id == "paper_a"

    def test_filter_non_papers(self) -> None:
        """
        Test that non-paper neighbors are filtered out.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)

        # Add a node "Adverse Event Z" connected to Protein X directly
        # This is a 1-hop neighbor that is NOT a paper.
        # It should be filtered out by the "neighbor.label == 'Paper'" check.
        client.nodes["ae_z"] = client.nodes["liver_failure"].model_copy()
        client.nodes["ae_z"].node_id = "ae_z"
        client.nodes["ae_z"].name = "Rash"

        client.edges.append({"source": "protein_x", "target": "ae_z"})

        request = SearchRequest(query="Protein X", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        # Should still be 1 paper (Paper A). Paper B is filtered (no AE). AE Z is filtered (not Paper).
        assert len(hits) == 1
        assert hits[0].doc_id == "paper_a"
