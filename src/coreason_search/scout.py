# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Dict, List, Optional, Union

from coreason_search.interfaces import BaseScout
from coreason_search.schemas import Hit


class MockScout(BaseScout):
    """
    Mock implementation of The Scout.
    Does not use heavy ML models.
    """

    def distill(self, query: Union[str, Dict[str, str]], hits: List[Hit]) -> List[Hit]:
        """
        Mock distillation: just copies original_text to distilled_text,
        maybe truncates it to simulate 'distillation'.
        """
        distilled_hits = []
        for hit in hits:
            # For the mock, we pretend we 'distilled' it by taking the first 50% of the characters
            # or just copying it if it's short.
            # This simulates the "removal of fluff".
            original_len = len(hit.original_text)
            # If empty, slicing gives empty
            keep_len = max(1, original_len // 2) if original_len > 0 else 0

            # Important: In a real implementation, we'd update distilled_text.
            # We must return a new Hit or modify the existing one.
            # Pydantic models are mutable by default unless configured otherwise.

            # Let's create a copy to avoid side effects on the input list if needed,
            # but for performance we might modify in place.
            # The interface says returns List[Hit].

            # We'll just modify the current hit instance for now or create a copy if we want to be functional.
            # Given Pydantic, copy() (v1) or model_copy() (v2) is good.
            new_hit = hit.model_copy()
            if original_len == 0:
                 new_hit.distilled_text = "..."
            else:
                 new_hit.distilled_text = hit.original_text[:keep_len] + "..."

            distilled_hits.append(new_hit)

        return distilled_hits


_scout_instance: Optional[BaseScout] = None


def get_scout() -> BaseScout:
    """Singleton factory for Scout."""
    global _scout_instance
    if _scout_instance is None:
        _scout_instance = MockScout()
    return _scout_instance


def reset_scout() -> None:
    """Reset singleton."""
    global _scout_instance
    _scout_instance = None
