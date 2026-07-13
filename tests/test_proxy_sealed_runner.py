from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg
from auctionlab.bids.xor import Bundle, XorAtomicBid, XorBid
from auctionlab.experiments.llm_comparison import proxy_sealed_result_to_row
from auctionlab.experiments.proxy_sealed_runner import (
    ProxySealedConfig,
    run_proxy_sealed_vcg_experiment,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.proxies.base import ElicitationEvent, ProxyStats, RefinementRecord
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def _upsert(bid: XorBid, bundle: Bundle, value: float) -> None:
    for idx, atom in enumerate(bid.atoms):
        if atom.bundle == bundle:
            bid.atoms[idx] = XorAtomicBid(bundle=bundle, value=value)
            return
    bid.atoms.append(XorAtomicBid(bundle=bundle, value=value))


@dataclass
class ScriptedSealedProxy:
    """A test double whose ``refine`` replaces an atom's value on cue."""

    bidder_id: str
    initial_atoms: list[XorAtomicBid]
    refined_value_by_bundle: dict[Bundle, float] = field(default_factory=dict)
    _bid: XorBid = field(init=False, repr=False)
    _stats: ProxyStats = field(default_factory=ProxyStats, init=False)

    def __post_init__(self) -> None:
        self._bid = XorBid(bidder_id=self.bidder_id, atoms=list(self.initial_atoms))

    def current_bid(self) -> XorBid:
        return self._bid

    def submit_bid(self) -> XorBid:
        return self._bid

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundle is None:
            return
        value = self.refined_value_by_bundle.get(event.bundle)
        if value is None:
            return
        self._stats.refinement_queries += 1
        _upsert(self._bid, event.bundle, value)

    def receive_provisional_feedback(self, event: ElicitationEvent) -> None:
        self.refine(event)

    def stats(self) -> ProxyStats:
        return self._stats


@dataclass
class RecordingScriptedSealedProxy(ScriptedSealedProxy):
    """``ScriptedSealedProxy`` that also tracks ``RefinementRecord``\\ s."""

    _records: list[RefinementRecord] = field(default_factory=list, init=False, repr=False)

    def refine(self, event: ElicitationEvent) -> None:
        if event.bundle is None:
            return
        new_value = self.refined_value_by_bundle.get(event.bundle)
        if new_value is None:
            return

        old_value = None
        for atom in self._bid.atoms:
            if atom.bundle == event.bundle:
                old_value = atom.value
                break

        self._stats.refinement_queries += 1
        _upsert(self._bid, event.bundle, new_value)
        self._records.append(
            RefinementRecord(
                bidder_id=self.bidder_id,
                mechanism=event.mechanism,
                event_type=event.event_type,
                round_idx=event.round_idx,
                bundle=event.bundle,
                old_value=old_value,
                new_value=new_value,
                reason=event.reason,
            )
        )

    def refinement_records(self) -> list[RefinementRecord]:
        return list(self._records)


def make_two_bidder_instance() -> AuctionInstance:
    return AuctionInstance(
        items=["A"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {frozenset({"A"}): 5.0},
            "i2": {frozenset({"A"}): 8.0},
        },
    )


def test_static_config_reproduces_sealed_vcg_on_initial_bids():
    instance = AuctionInstance(
        items=["A", "B"],
        bidder_ids=["i1", "i2"],
        valuations={
            "i1": {
                frozenset({"A"}): 10.0,
                frozenset({"B"}): 5.0,
            },
            "i2": {
                frozenset({"A"}): 4.0,
                frozenset({"B"}): 9.0,
            },
        },
    )
    proxies = [
        FullInfoAuctionProxy(bidder_id="i1", instance=instance, initial="all_atoms"),
        FullInfoAuctionProxy(bidder_id="i2", instance=instance, initial="all_atoms"),
    ]

    result = run_proxy_sealed_vcg_experiment(
        instance,
        proxies,
        ProxySealedConfig(),
    )

    expected = run_sealed_xor_vcg(
        items=instance.items,
        bids=[proxy.submit_bid() for proxy in proxies],
    )

    assert result.mechanism == "proxy_sealed_vcg_static"
    assert result.allocation == expected.allocation
    assert result.welfare == expected.welfare
    assert result.payments == expected.payments
    assert result.metadata["elicitation_rounds"] == 0
    assert result.metadata["feedback_rule"] == "none"
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 0,
        "i2": 0,
    }


def test_elicitation_round_sends_feedback_and_improves_allocation():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    # Static: i2 wins (8 > 5).
    static_result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(),
    )
    assert static_result.allocation["i2"] == frozenset({"A"})
    assert static_result.allocation["i1"] == frozenset()

    # Reset for the elicited run.
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    elicited_result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=1,
        ),
    )

    assert elicited_result.mechanism == (
        "proxy_sealed_vcg_elicited_lost_interested_bundle_1"
    )
    # After refinement, i1's value for A (20) beats i2's (8).
    assert elicited_result.allocation["i1"] == frozenset({"A"})
    assert elicited_result.allocation["i2"] == frozenset()
    assert elicited_result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 1,
        "i2": 0,
    }
    assert elicited_result.metadata["feedback_rule"] == "lost_interested_bundle"
    assert elicited_result.metadata["elicitation_rounds"] == 1


def test_max_refinements_per_bidder_zero_is_unlimited():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=3,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=0,
        ),
    )

    # A cap of 0 means unlimited, so i1's refinement goes through and i1
    # ends up winning with its corrected value of 20.
    assert result.allocation["i1"] == frozenset({"A"})
    assert result.metadata["refinement_query_count_by_bidder"] == {
        "i1": 1,
        "i2": 0,
    }


def test_proxy_sealed_row_exports_final_updated_bid_and_refinement_records():
    instance = make_two_bidder_instance()
    i1 = RecordingScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
        refined_value_by_bundle={frozenset({"A"}): 20.0},
    )
    i2 = ScriptedSealedProxy(
        bidder_id="i2",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=8.0)],
    )

    result = run_proxy_sealed_vcg_experiment(
        instance,
        [i1, i2],
        ProxySealedConfig(
            elicitation_rounds=1,
            feedback_rule="lost_interested_bundle",
            max_refinements_per_bidder=1,
        ),
    )

    # The proxy's own bid is updated in place.
    assert i1.current_bid().value_of(frozenset({"A"})) == 20.0
    assert i1.submit_bid().value_of(frozenset({"A"})) == 20.0

    records = result.metadata["refinement_records_by_bidder"]["i1"]
    assert len(records) == 1
    assert records[0].old_value == 5.0
    assert records[0].new_value == 20.0

    row = proxy_sealed_result_to_row(
        instance_name="two_bidder",
        instance=instance,
        result=result,
    )

    # Initial bid snapshot is unaffected by the later refinement.
    assert "i1={[A]:5.0}" in row["initial_bids"]
    assert "i1={[A]:5.0}" in row["initial_reported_bids"]
    # Final bid reflects the post-refinement value.
    assert "i1={[A]:20.0}" in row["final_bids"]
    assert "i1={[A]:20.0}" in row["final_reported_bids"]
    assert "i1:[A] 5.0->20.0" in row["refinement_records"]


def test_proxies_must_match_instance_bidder_ids():
    instance = make_two_bidder_instance()
    i1 = ScriptedSealedProxy(
        bidder_id="i1",
        initial_atoms=[XorAtomicBid(bundle=frozenset({"A"}), value=5.0)],
    )

    with pytest.raises(ValueError, match="proxies must contain"):
        run_proxy_sealed_vcg_experiment(instance, [i1], ProxySealedConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elicitation_rounds": -1},
        {"feedback_rule": "bogus"},
        {"max_refinements_per_bidder": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ProxySealedConfig(**kwargs)
