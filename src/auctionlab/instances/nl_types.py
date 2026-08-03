"""NaturalLanguageAuctionScenario dataclass.

Kept in its own module so both nl_scenarios (which populates it with
hand-written scenarios) and structured (which generates scenarios
programmatically) can import it without creating a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from auctionlab.instances.base import AuctionInstance


@dataclass(frozen=True)
class NaturalLanguageAuctionScenario:
    """An auction scenario with natural-language person disclosures.

    Attributes
    ----------
    name:
        Unique scenario identifier.
    instance:
        The underlying auction instance (items, bidder IDs, valuation table).
    scenario_description:
        A brief description of the auction context shown to LLM proxies.
    item_descriptions:
        Per-item natural-language descriptions shown to LLM proxies.
    person_seeds:
        Per-bidder natural-language disclosures used to initialise each
        :class:`~auctionlab.llm.person_simulator.LlmPersonSimulator`. For
        structured scenarios these are brief and qualitative; exact values
        remain in ``instance.valuations``.
    seed_type:
        ``"explicit"`` — seed enumerates bundle values explicitly;
        ``"implicit"`` — seed describes preferences without exact values;
        ``"structured"`` — seed generated from a latent preference profile.
    candidate_bundles_by_bidder:
        Optional per-bidder candidate bundle lists. ``None`` for structured
        scenarios (proxies derive candidates from the interest map or
        generic enumeration).
    metadata:
        Arbitrary key–value metadata for diagnostics and logging.
    """

    name: str
    instance: AuctionInstance
    scenario_description: str
    item_descriptions: dict[str, str]
    person_seeds: dict[str, str]
    seed_type: str = "explicit"
    candidate_bundles_by_bidder: dict[str, list[frozenset[str]]] | None = None
    metadata: dict = field(default_factory=dict)
