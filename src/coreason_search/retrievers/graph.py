# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import List, Set

from coreason_search.graph_client import GraphNode, get_graph_client
from coreason_search.interfaces import BaseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.utils.common import extract_query_text
from coreason_search.utils.logger import logger


class GraphRetriever(BaseRetriever):
    """
    Graph Retriever Strategy.
    Performs 2-hop neighbor expansion:
    Query -> Papers -> AdverseEvents
    """

    def __init__(self) -> None:
        self.client = get_graph_client()

    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """
        Execute Graph Retrieval.
        1. Search for nodes matching the query.
        2. Expand 1-hop to find connected "Paper" nodes.
        3. Expand 2-hop to find connected "AdverseEvent" nodes.
        4. Return Papers that bridge Query -> Paper -> AdverseEvent.
        """
        query_text = extract_query_text(request.query)

        # 1. Identify start nodes (Entity Linking step simplified)
        start_nodes = self.client.search_nodes(query_text)

        if not start_nodes:
            logger.info(f"No graph nodes found for query: {query_text}")
            return []

        hits: List[Hit] = []
        seen_ids: Set[str] = set()

        for node in start_nodes:
            self._process_start_node(node, hits, seen_ids)

        return hits[: request.top_k]

    def _process_start_node(self, node: GraphNode, hits: List[Hit], seen_ids: Set[str]) -> None:
        """
        Expand from a start node to find connected Papers and validate them.

        Args:
            node: The starting GraphNode (from query).
            hits: List to append valid hits to.
            seen_ids: Set of doc_ids to prevent duplicates.
        """
        # 2. Traversal: Get neighbors (Papers)
        neighbors = self.client.get_neighbors(node.node_id)

        for neighbor in neighbors:
            if neighbor.label == "Paper" and neighbor.node_id not in seen_ids:
                self._validate_and_add_paper(neighbor, hits, seen_ids)

    def _validate_and_add_paper(self, paper_node: GraphNode, hits: List[Hit], seen_ids: Set[str]) -> None:
        """
        Check if a candidate paper connects to an Adverse Event, and if so, add it.

        Args:
            paper_node: The candidate Paper node.
            hits: List to append hits to.
            seen_ids: Set of seen doc_ids.
        """
        # 3. Validation: Does this paper connect to an AdverseEvent?
        # Perform 2nd hop
        paper_neighbors = self.client.get_neighbors(paper_node.node_id)

        # Identify Adverse Events (Use set for deduplication)
        adverse_events_set = {n.name for n in paper_neighbors if n.label == "AdverseEvent"}

        if adverse_events_set:
            seen_ids.add(paper_node.node_id)
            hits.append(self._create_hit(paper_node, adverse_events_set))

    def _create_hit(self, paper_node: GraphNode, adverse_events_set: Set[str]) -> Hit:
        """
        Construct a Hit object from a Paper node and its connected adverse events.

        Args:
            paper_node: The GraphNode representing the paper.
            adverse_events_set: Set of names of connected adverse events.

        Returns:
            Hit: The populated Hit object.
        """
        content = str(paper_node.properties.get("content", ""))

        # Enrich Metadata
        # Copy properties to avoid modifying cached/original object
        metadata = paper_node.properties.copy()
        # Sort for deterministic output
        metadata["connected_adverse_events"] = sorted(list(adverse_events_set))

        return Hit(
            doc_id=paper_node.node_id,
            content=content,
            original_text=content,
            distilled_text="",
            score=1.0,
            source_strategy=RetrieverType.GRAPH_NEIGHBOR.value,
            metadata=metadata,
        )
