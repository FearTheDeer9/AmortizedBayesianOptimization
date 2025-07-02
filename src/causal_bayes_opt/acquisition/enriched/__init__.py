"""
Enriched context architecture for acquisition policy.

This module implements the enhanced context architecture that moves all
context information into the transformer input rather than concatenating
features after transformer processing.
"""

from .state_enrichment import EnrichedHistoryBuilder
from .enriched_policy import EnrichedAttentionEncoder
from .policy_heads import SimplifiedPolicyHeads

__all__ = [
    "EnrichedHistoryBuilder",
    "EnrichedAttentionEncoder", 
    "SimplifiedPolicyHeads",
]