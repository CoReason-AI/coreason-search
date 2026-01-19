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


class TestGraphTraversal2Hop:
    def setup_method(self) -> None:
        reset_graph_client()

    def test_filter_papers_without_adverse_events(self) -> None:
        """
        Test that papers connected to the entity are ONLY returned if they
        ALSO connect to an AdverseEvent (2-hop).

        Mock Data Setup:
        - Protein X -> Paper A (Connects to 'Liver Failure' [AdverseEvent])
        - Protein X -> Paper B (No AdverseEvent connection)

        Expectation: Only Paper A is returned.
        """
        retriever = GraphRetriever()

        # Verify Mock Data assumptions
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        # Ensure 'paper_a' connects to 'liver_failure'
        # Ensure 'paper_b' does not connect to any AdverseEvent

        request = SearchRequest(
            query="Protein X",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )

        hits = retriever.retrieve(request)

        # Assertion: Should only return 1 hit (Paper A)
        assert len(hits) == 1
        assert hits[0].doc_id == "paper_a"
        assert "liver failure" in hits[0].content

        # Metadata Enrichment Check
        # Expect 'connected_adverse_events' to be a list containing "Liver Failure"
        assert "connected_adverse_events" in hits[0].metadata
        ae_list = hits[0].metadata["connected_adverse_events"]
        assert isinstance(ae_list, list)
        assert "Liver Failure" in ae_list

    def test_paper_with_non_ae_neighbor(self) -> None:
        """
        Test that a paper connecting to another node (not AdverseEvent)
        is still filtered out.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)

        # Add 'Paper C' connected to Protein X
        # Add 'Protein Y' connected to Paper C
        # Paper C -> Protein Y (No AdverseEvent)

        # 1. Create Nodes
        from coreason_search.graph_client import GraphNode

        paper_c = GraphNode(node_id="paper_c", label="Paper", name="Study C", properties={"content": "Content C"})
        protein_y = GraphNode(
            node_id="protein_y",
            label="Protein",
            name="Protein Y",
        )

        client.nodes["paper_c"] = paper_c
        client.nodes["protein_y"] = protein_y

        # 2. Add Edges
        client.edges.append({"source": "protein_x", "target": "paper_c"})
        client.edges.append({"source": "paper_c", "target": "protein_y"})

        request = SearchRequest(query="Protein X", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        # Expectation: Only Paper A returned.
        # Paper B (disconnected) -> Filtered
        # Paper C (connected to Protein Y) -> Filtered
        ids = [h.doc_id for h in hits]
        assert "paper_a" in ids
        assert "paper_b" not in ids
        assert "paper_c" not in ids
        assert len(hits) == 1
