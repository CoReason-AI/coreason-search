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


class TestGraphRetrieverComplex:
    def setup_method(self) -> None:
        reset_graph_client()

    def _setup_complex_graph(self, client: MockGraphClient) -> None:
        """
        Helper to inject complex data into the mock client.
        Structure:
          - "Term A1" (Protein) -> "Paper Common"
          - "Term A2" (Protein) -> "Paper Common", "Paper Unique"
          - "Term B" (Protein) -> (no neighbors)
          - "Term C" (Protein) -> "Paper Safe", "Event Danger" (AdverseEvent)
          - "Paper Broken" (Paper) -> No content property

        Updated for 2-Hop Requirements:
        Papers must connect to an "AdverseEvent" to be returned.
        """
        # Clear existing for clean state
        client.nodes = {}
        client.edges = []

        # Nodes
        client.nodes["a1"] = GraphNode(node_id="a1", label="Protein", name="Term A1")
        client.nodes["a2"] = GraphNode(node_id="a2", label="Protein", name="Term A2")
        client.nodes["b"] = GraphNode(node_id="b", label="Protein", name="Term B")
        client.nodes["c"] = GraphNode(node_id="c", label="Protein", name="Term C")

        client.nodes["p_common"] = GraphNode(
            node_id="p_common",
            label="Paper",
            name="Common Paper",
            properties={"content": "Common content"},
        )
        client.nodes["p_unique"] = GraphNode(
            node_id="p_unique",
            label="Paper",
            name="Unique Paper",
            properties={"content": "Unique content"},
        )
        client.nodes["p_broken"] = GraphNode(
            node_id="p_broken",
            label="Paper",
            name="Broken Paper",
            properties={},  # Missing content
        )
        client.nodes["p_safe"] = GraphNode(
            node_id="p_safe",
            label="Paper",
            name="Safe Paper",
            properties={"content": "Safe content"},
        )
        client.nodes["e_danger"] = GraphNode(
            node_id="e_danger",
            label="AdverseEvent",
            name="Danger Event",
            properties={"content": "Bad things"},
        )
        client.nodes["e_toxicity"] = GraphNode(
            node_id="e_toxicity",
            label="AdverseEvent",
            name="Toxicity Event",
            properties={"content": "Toxic"},
        )

        # Edges
        # A1 -> Common
        client.edges.append({"source": "a1", "target": "p_common"})
        # A2 -> Common, Unique
        client.edges.append({"source": "a2", "target": "p_common"})
        client.edges.append({"source": "a2", "target": "p_unique"})

        # 2-Hop Connections (Papers -> AdverseEvents)
        # Connect Common to Danger
        client.edges.append({"source": "p_common", "target": "e_danger"})
        # Connect Unique to Toxicity
        client.edges.append({"source": "p_unique", "target": "e_toxicity"})

        # B -> None
        # C -> Safe, Danger
        client.edges.append({"source": "c", "target": "p_safe"})
        client.edges.append({"source": "c", "target": "e_danger"})

        # Connect Safe to Danger (so p_safe is returned)
        client.edges.append({"source": "p_safe", "target": "e_danger"})

        # Setup lookup for A1/A2 to work with substring "Term A"
        # The mock client implementation uses `if query_lower in node.name.lower()`
        # So "Term A" will match "Term A1" and "Term A2".

    def test_multi_node_deduplication(self) -> None:
        """
        Scenario 1: Multi-Node Expansion & Deduplication.
        Query "Term A" matches A1 and A2.
        A1 -> P_Common (Connects to Danger)
        A2 -> P_Common, P_Unique (Connects to Toxicity)
        Expected: [P_Common, P_Unique] (P_Common appearing once).
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._setup_complex_graph(client)

        request = SearchRequest(query="Term A", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 2
        doc_ids = sorted([h.doc_id for h in hits])
        assert doc_ids == ["p_common", "p_unique"]

    def test_zero_neighbors(self) -> None:
        """
        Scenario 3: Zero Neighbors.
        Query "Term B" matches matches Node B, which has no edges.
        Expected: Empty list.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._setup_complex_graph(client)

        request = SearchRequest(query="Term B", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 0

    def test_strict_label_filtering(self) -> None:
        """
        Scenario 4: Strict Label Filtering.
        Query "Term C" matches matches Node C.
        C -> P_Safe (Paper), E_Danger (AdverseEvent).
        P_Safe -> E_Danger (2-hop connection).
        E_Danger is filtered (not Paper).
        Expected: [P_Safe].
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._setup_complex_graph(client)

        request = SearchRequest(query="Term C", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "p_safe"

    def test_missing_data_robustness(self) -> None:
        """
        Scenario 2: Robustness to Missing Data.
        Inject a node that connects to "Broken Paper" (no content).
        Connect Broken Paper to an Adverse Event to satisfy filter.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._setup_complex_graph(client)

        # Connect B to Broken manually
        client.edges.append({"source": "b", "target": "p_broken"})
        # Connect Broken to Danger (2-hop)
        client.edges.append({"source": "p_broken", "target": "e_danger"})

        request = SearchRequest(query="Term B", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].doc_id == "p_broken"
        assert hits[0].content == ""  # Graceful fallback to empty string

    def test_special_characters(self) -> None:
        """
        Scenario 5: Special Characters.
        Query with chars that might break regex if used incorrectly, though Mock uses string 'in'.
        """
        retriever = GraphRetriever()
        client = retriever.client
        assert isinstance(client, MockGraphClient)
        self._setup_complex_graph(client)

        # Rename A1 to include special chars
        client.nodes["a1"].name = "Protein #1 (Complex)"

        request = SearchRequest(query="Protein #1", strategies=[RetrieverType.GRAPH_NEIGHBOR])
        hits = retriever.retrieve(request)

        # Should match A1 -> P_Common (Connects to Danger)
        assert len(hits) == 1
        assert hits[0].doc_id == "p_common"
