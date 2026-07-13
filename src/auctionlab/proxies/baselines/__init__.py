"""Diagnostic and legacy proxy baselines.

These proxies are kept for reproducibility and literature comparison but are
not part of the main modular LLM proxy architecture:

- :class:`~auctionlab.proxies.baselines.dnf_learning.DnfLearningProxy`
  (ωxor) — proper-learning baseline that learns an exact XOR bid from
  demand and value queries without any LLM inference.

- :class:`~auctionlab.proxies.baselines.hybrid.HybridProxy`
  (ωh) — uses the modular LLM proxy for the first ``alpha`` refinement
  events, then switches to the DNF baseline.

- :class:`~auctionlab.proxies.baselines.llm_ceca.Vd1CecaProxy` (ωvd1),
  :class:`~auctionlab.proxies.baselines.llm_ceca.Vd2CecaProxy` (ωvd2),
  :class:`~auctionlab.proxies.baselines.llm_ceca.NvdCecaProxy` (ωnvd) —
  legacy CECA-specific action-choice proxies from the original paper
  taxonomy, superseded by the modular LLM proxy's ``ceca_step``
  implementation.
"""

from __future__ import annotations

from auctionlab.proxies.baselines.dnf_learning import DnfLearningProxy
from auctionlab.proxies.baselines.hybrid import HybridProxy
from auctionlab.proxies.baselines.llm_ceca import (
    AllBundlesScope,
    BundleScope,
    InterestMapScope,
    NvdCecaProxy,
    SizeLimitedScope,
    Vd1CecaProxy,
    Vd2CecaProxy,
)

__all__ = [
    "DnfLearningProxy",
    "HybridProxy",
    "AllBundlesScope",
    "BundleScope",
    "InterestMapScope",
    "NvdCecaProxy",
    "SizeLimitedScope",
    "Vd1CecaProxy",
    "Vd2CecaProxy",
]
