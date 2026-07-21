"""Diagnostic proxy baselines for literature comparison and ablations.

These proxies are kept for reproducibility but are not part of the main
modular LLM proxy architecture:

- :class:`~auctionlab.proxies.baselines.dnf_learning.DnfLearningProxy`
  (ωxor) — proper-learning baseline that learns an exact XOR bid from
  demand and value queries without any LLM inference.

- :class:`~auctionlab.proxies.baselines.hybrid.HybridProxy`
  (ωh) — uses the modular LLM proxy for the first ``alpha`` refinement
  events, then switches to the DNF baseline.
"""

from __future__ import annotations

from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy
from auctionlab.proxies.baselines.hybrid import HybridProxy

__all__ = [
    "DnfLearningProxy",
    "HybridProxy",
]
