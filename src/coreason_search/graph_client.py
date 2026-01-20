# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Represents a node in the Knowledge Graph.

    Attributes:
        node_id: The unique identifier of the node.
        label: The label of the node (e.g., "Protein", "Paper").
        name: The human-readable name of the node.
        properties: Additional properties of the node.
    """

    node_id: str
    label: str  # e.g., "Protein", "Paper", "AdverseEvent"
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class BaseGraphClient(ABC):
    """Abstract base class for the Graph Nexus client."""

    @abstractmethod
    def search_nodes(self, query: str, limit: int = 5) -> List[GraphNode]:
        """Search for nodes matching the query string.

        Args:
            query: The search string (e.g., "Protein X").
            limit: Max number of nodes to return. Defaults to 5.

        Returns:
            List[GraphNode]: List of matching nodes.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_neighbors(self, node_id: str, hop_depth: int = 1) -> List[GraphNode]:
        """Get neighbors of a node.

        Args:
            node_id: The ID of the start node.
            hop_depth: Number of hops. Defaults to 1.

        Returns:
            List[GraphNode]: List of neighbor nodes.
        """
        pass  # pragma: no cover


class MockGraphClient(BaseGraphClient):
    """Mock implementation of Graph Client.

    Simulates a small knowledge graph for testing and development.
    """

    def __init__(self) -> None:
        """Initialize the mock graph client with dummy data."""
        # Define some mock data
        # "Protein X" -> "Paper A" (discusses mechanism)
        # "Paper A" -> "Adverse Event Y" (mentioned in paper)
        self.nodes: Dict[str, GraphNode] = {
            "protein_x": GraphNode(
                node_id="protein_x",
                label="Protein",
                name="Protein X",
                properties={"description": "Target protein"},
            ),
            "paper_a": GraphNode(
                node_id="paper_a",
                label="Paper",
                name="Study on Protein X",
                properties={"content": "This paper discusses Protein X and liver failure.", "year": 2024},
            ),
            "paper_b": GraphNode(
                node_id="paper_b",
                label="Paper",
                name="Another Study",
                properties={"content": "Protein X is safe.", "year": 2023},
            ),
            "liver_failure": GraphNode(
                node_id="liver_failure",
                label="AdverseEvent",
                name="Liver Failure",
                properties={},
            ),
        }

        self.edges: List[Dict[str, str]] = [
            {"source": "protein_x", "target": "paper_a"},
            {"source": "protein_x", "target": "paper_b"},
            {"source": "paper_a", "target": "liver_failure"},
        ]

    def search_nodes(self, query: str, limit: int = 5) -> List[GraphNode]:
        """Simple substring match on name.

        Args:
            query: The search string.
            limit: Maximum number of results.

        Returns:
            List[GraphNode]: Matching nodes.
        """
        query_lower = query.lower()
        matches = []
        for node in self.nodes.values():
            if query_lower in node.name.lower():
                matches.append(node)
        return matches[:limit]

    def get_neighbors(self, node_id: str, hop_depth: int = 1) -> List[GraphNode]:
        """Get 1-hop neighbors using edge list.

        Ignores hop_depth > 1 for this mock to stay atomic/simple.

        Args:
            node_id: The ID of the start node.
            hop_depth: Number of hops. Defaults to 1.

        Returns:
            List[GraphNode]: Neighboring nodes.
        """
        if hop_depth != 1:
            # For simplicity, we only support 1-hop in mock currently
            pass

        neighbors = []
        # Check outgoing
        for edge in self.edges:
            if edge["source"] == node_id:
                if edge["target"] in self.nodes:
                    neighbors.append(self.nodes[edge["target"]])
            elif edge["target"] == node_id:
                if edge["source"] in self.nodes:
                    neighbors.append(self.nodes[edge["source"]])
        return neighbors


@lru_cache(maxsize=32)
def get_graph_client() -> BaseGraphClient:
    """Singleton factory for Graph Client.

    Returns:
        BaseGraphClient: An instance of the graph client.
    """
    return MockGraphClient()


def reset_graph_client() -> None:
    """Reset singleton (clear cache)."""
    get_graph_client.cache_clear()
