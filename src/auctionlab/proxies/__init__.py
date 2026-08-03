"""Proxy-mediated preference elicitation for combinatorial auctions.

Main architecture
-----------------
The primary proxy is the **modular LLM proxy**, composed of two classes from
:mod:`auctionlab.llm.proxies`:

- :class:`~auctionlab.llm.proxies.LlmInferredXorProxy` — infers XOR bids
  using an LLM-backed :class:`~auctionlab.llm.person_simulator.LlmPersonSimulator`.
- :class:`~auctionlab.llm.proxies.LlmAuctionProxyAdapter` — wraps an
  ``LlmInferredXorProxy`` and exposes the full protocol for sealed and clock
  mechanisms.

Top-level modules in this package
----------------------------------
- :mod:`~auctionlab.proxies.base` — shared data types
  (:class:`~auctionlab.proxies.base.ElicitationEvent`,
  :class:`~auctionlab.proxies.base.RefinementRecord`,
  :class:`~auctionlab.proxies.base.ProxyStats`).
- :mod:`~auctionlab.proxies.elicitation` — elicitation helpers and
  :func:`~auctionlab.proxies.elicitation.replay_elicitation`.
- :mod:`~auctionlab.proxies.full_info` — full-information benchmark proxy
  used by evaluation and fast mechanism tests.
"""

from __future__ import annotations
