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
        query_text = request.query
        if isinstance(query_text, dict):
            # Graph search usually expects an entity name string.
            # Convert or fallback.
            query_text = " ".join(str(v) for v in query_text.values())

        # 1. Identify start nodes (Entity Linking step simplified)
        start_nodes = self.client.search_nodes(str(query_text))

        if not start_nodes:
            logger.info(f"No graph nodes found for query: {query_text}")
            return []

        hits: List[Hit] = []
        seen_ids = set()

        for node in start_nodes:
            # 2. Traversal: Get neighbors (Papers)
            neighbors = self.client.get_neighbors(node.node_id)

            for neighbor in neighbors:
                if neighbor.label == "Paper" and neighbor.node_id not in seen_ids:
                    # 3. Validation: Does this paper connect to an AdverseEvent?
                    # Perform 2nd hop
                    paper_neighbors = self.client.get_neighbors(neighbor.node_id)
                    has_adverse_event = any(n.label == "AdverseEvent" for n in paper_neighbors)

                    if has_adverse_event:
                        seen_ids.add(neighbor.node_id)
                        content = neighbor.properties.get("content", "")

                        hits.append(
                            Hit(
                                doc_id=neighbor.node_id,
                                content=content,
                                original_text=content,
                                distilled_text="",
                                score=1.0,
                                source_strategy=RetrieverType.GRAPH_NEIGHBOR.value,
                                metadata=neighbor.properties,
                            )
                        )

        return hits[: request.top_k]
