# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import List

from coreason_search.graph_client import get_graph_client
from coreason_search.interfaces import BaseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.utils.logger import logger


class GraphRetriever(BaseRetriever):
    """
    Graph Retriever Strategy.
    Performs 1-hop neighbor expansion from found entities to documents.
    """

    def __init__(self) -> None:
        self.client = get_graph_client()

    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """
        Execute Graph Retrieval.
        1. Search for nodes matching the query.
        2. Expand 1-hop to find connected "Paper" nodes.
        3. Convert to Hits.
        """
        query_text = request.query
        if isinstance(query_text, dict):
            # Graph search usually expects an entity name string.
            # Convert or fallback.
            query_text = " ".join(str(v) for v in query_text.values())

        # 1. Identify start nodes (Entity Linking step simplified)
        # In a real system, we'd use an Entity Linker. Here we just search the graph.
        start_nodes = self.client.search_nodes(str(query_text))

        if not start_nodes:
            logger.info(f"No graph nodes found for query: {query_text}")
            return []

        hits: List[Hit] = []
        seen_ids = set()

        for node in start_nodes:
            # 2. Traversal: Get neighbors
            # We want "Papers".
            neighbors = self.client.get_neighbors(node.node_id)

            for neighbor in neighbors:
                # Filter: Only return "Paper" nodes as hits?
                # PRD: "Retrieve Node(Protein X) -> All connected Papers -> All connected Adverse Events."
                # If the user asks for papers about Protein X, we return Paper neighbors.
                # If the user asks for generic info, maybe we return everything?
                # Usually Search Engine returns Documents (Papers).
                # We'll assume we filter for label="Paper" or assume the graph structure puts text in papers.
                # The mock has label="Paper".

                if neighbor.label == "Paper" and neighbor.node_id not in seen_ids:
                    seen_ids.add(neighbor.node_id)

                    # Map GraphNode to Hit
                    content = neighbor.properties.get("content", "")

                    hits.append(
                        Hit(
                            doc_id=neighbor.node_id,
                            content=content,
                            original_text=content,
                            distilled_text="",
                            score=1.0,  # Graph traversal implies high relevance (binary connection)
                            source_strategy=RetrieverType.GRAPH_NEIGHBOR.value,
                            metadata=neighbor.properties,
                        )
                    )

        return hits[: request.top_k]
