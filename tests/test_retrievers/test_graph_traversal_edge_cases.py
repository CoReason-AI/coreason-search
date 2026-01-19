# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from coreason_search.graph_client import GraphNode, MockGraphClient, reset_graph_client
from coreason_search.retrievers.graph import GraphRetriever
from coreason_search.schemas import RetrieverType, SearchRequest


class TestGraphTraversalEdgeCases:
    def setup_method(self) -> None:
        reset_graph_client()

    def _clear_graph(self, client: MockGraphClient) -> None:
        """Helper to clear default mock data."""
        client.nodes = {}
        client.edges = []

    def test_reverse_lookup(self) -> None:
        """
        Test "Reverse Lookup": Querying an AdverseEvent node should return
        the Paper connected to it, provided the Paper validates the logic
        (which it does, because it connects to the AE).
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        # Use existing mock data, which has Protein X -> Paper A -> Liver Failure

        request = SearchRequest(
            query="Liver Failure",
            strategies=[RetrieverType.GRAPH_NEIGHBOR],
        )

        hits = retriever.retrieve(request)

        # Should find Paper A
        assert len(hits) == 1
        assert hits[0].doc_id == "paper_a"
        # Since logic checks if paper connects to AE, and Paper A connects to Liver Failure (the query node),
        # it works.

    def test_multiple_adverse_events(self) -> None:
        """
        Test a Paper connected to multiple Adverse Events.
        It should be returned exactly once (deduplicated).
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        # Create Paper M (Multi)
        client.nodes["paper_m"] = GraphNode(
            node_id="paper_m", label="Paper", name="Multi AE Paper", properties={"content": "M"}
        )
        # Create Entity Start
        client.nodes["start_m"] = GraphNode(node_id="start_m", label="Protein", name="Start M")
        # Create AE 1 and AE 2
        client.nodes["ae_1"] = GraphNode(node_id="ae_1", label="AdverseEvent", name="AE 1")
        client.nodes["ae_2"] = GraphNode(node_id="ae_2", label="AdverseEvent", name="AE 2")

        # Edges
        client.edges.append({"source": "start_m", "target": "paper_m"})
        client.edges.append({"source": "paper_m", "target": "ae_1"})
        client.edges.append({"source": "paper_m", "target": "ae_2"})

        request = SearchRequest(query="Start M", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "paper_m"

        # New assertion: Check metadata contains both AEs
        aes = hits[0].metadata["connected_adverse_events"]
        assert "AE 1" in aes
        assert "AE 2" in aes
        assert len(aes) == 2

    def test_duplicate_adverse_events(self) -> None:
        """
        Test a Paper connected to Adverse Events with duplicate names/nodes.
        Metadata should deduplicate.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        client.nodes["paper_d"] = GraphNode(
            node_id="paper_d", label="Paper", name="Dup Paper", properties={"content": "D"}
        )
        client.nodes["start_d"] = GraphNode(node_id="start_d", label="Protein", name="Start D")

        # AE 1 and AE 1_dup (same name)
        client.nodes["ae_1"] = GraphNode(node_id="ae_1", label="AdverseEvent", name="Nausea")
        client.nodes["ae_1_dup"] = GraphNode(node_id="ae_1_dup", label="AdverseEvent", name="Nausea")

        client.edges.append({"source": "start_d", "target": "paper_d"})
        client.edges.append({"source": "paper_d", "target": "ae_1"})
        client.edges.append({"source": "paper_d", "target": "ae_1_dup"})

        request = SearchRequest(query="Start D", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        aes = hits[0].metadata["connected_adverse_events"]
        # Should be deduplicated by name
        assert len(aes) == 1
        assert aes[0] == "Nausea"

    def test_mixed_neighbors(self) -> None:
        """
        Test a Paper with mixed neighbors: Protein, Paper, AdverseEvent.
        Logic should filter correctly and find the AE.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        # Paper X -> Protein Y (noise)
        # Paper X -> Paper Z (noise)
        # Paper X -> AE (Signal)

        client.nodes["paper_x"] = GraphNode(
            node_id="paper_x", label="Paper", name="Mixed X", properties={"content": "X"}
        )
        client.nodes["protein_y"] = GraphNode(node_id="protein_y", label="Protein", name="Y")
        client.nodes["paper_z"] = GraphNode(node_id="paper_z", label="Paper", name="Z", properties={"content": "Z"})
        client.nodes["ae_signal"] = GraphNode(node_id="ae_signal", label="AdverseEvent", name="Signal")
        client.nodes["start_x"] = GraphNode(node_id="start_x", label="Protein", name="Start X")

        # Connect Start -> Paper X
        client.edges.append({"source": "start_x", "target": "paper_x"})

        # Connect Paper X to others
        client.edges.append({"source": "paper_x", "target": "protein_y"})
        client.edges.append({"source": "paper_x", "target": "paper_z"})
        client.edges.append({"source": "paper_x", "target": "ae_signal"})

        request = SearchRequest(query="Start X", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "paper_x"

    def test_self_referential_loops(self) -> None:
        """
        Test graph cycles (Paper connects to itself).
        Should not crash.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        client.nodes["paper_loop"] = GraphNode(
            node_id="paper_loop", label="Paper", name="Loop", properties={"content": "L"}
        )
        client.nodes["start_l"] = GraphNode(node_id="start_l", label="Protein", name="Start L")
        client.nodes["ae_l"] = GraphNode(node_id="ae_l", label="AdverseEvent", name="AE L")

        # Edges
        client.edges.append({"source": "start_l", "target": "paper_loop"})
        client.edges.append({"source": "paper_loop", "target": "ae_l"})
        # Self loop
        client.edges.append({"source": "paper_loop", "target": "paper_loop"})

        request = SearchRequest(query="Start L", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "paper_loop"

    def test_chain_of_papers_fail(self) -> None:
        """
        Test that logic does NOT traverse Paper -> Paper -> AE.
        It is strictly Entity -> Paper -> AE.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        # Unique names to avoid accidental matches
        client.nodes["start_chain"] = GraphNode(node_id="start_chain", label="Protein", name="StartChain")
        client.nodes["paper_a"] = GraphNode(
            node_id="paper_a", label="Paper", name="PaperA", properties={"content": "A"}
        )
        client.nodes["paper_b"] = GraphNode(
            node_id="paper_b", label="Paper", name="PaperB", properties={"content": "B"}
        )
        client.nodes["ae_end"] = GraphNode(node_id="ae_end", label="AdverseEvent", name="AdverseEnd")

        # Start -> A -> B -> AE
        client.edges.append({"source": "start_chain", "target": "paper_a"})
        client.edges.append({"source": "paper_a", "target": "paper_b"})
        client.edges.append({"source": "paper_b", "target": "ae_end"})

        # Query "StartChain" only matches start_chain
        request = SearchRequest(query="StartChain", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        # Expect 0.
        assert len(hits) == 0

    def test_cyclic_adverse_event(self) -> None:
        """
        Test case where Query is an AE, connects to Paper, which connects back to the same AE.
        Query "Nausea" -> matches Node "Nausea" (AE).
        Nausea -> Paper A.
        Paper A -> Nausea.
        Should return Paper A with "Nausea" in metadata.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._clear_graph(client)

        client.nodes["nausea"] = GraphNode(node_id="nausea", label="AdverseEvent", name="Nausea")
        client.nodes["paper_cyc"] = GraphNode(
            node_id="paper_cyc", label="Paper", name="Paper Cyclic", properties={"content": "C"}
        )

        # Edge from Nausea to Paper (Paper discusses Nausea)
        client.edges.append({"source": "nausea", "target": "paper_cyc"})
        # Edge from Paper to Nausea (Paper identifies Nausea as AE)
        client.edges.append({"source": "paper_cyc", "target": "nausea"})

        request = SearchRequest(query="Nausea", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "paper_cyc"
        assert "Nausea" in hits[0].metadata["connected_adverse_events"]
