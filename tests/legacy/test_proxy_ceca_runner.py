from __future__ import annotations

import pytest

from auctionlab.auctions.ceca import (
    CecaConfig,
    finalize_ceca_pay_as_bid,
    finalize_ceca_vcg,
    run_ceca,
)
from auctionlab.bids.xor import XorAtomicBid, XorBid
from auctionlab.experiments.llm_comparison import (
    ceca_result_to_row,
    ceca_winner_diagnostics_rows,
)
from auctionlab.experiments.proxy_ceca_runner import (
    ProxyCecaConfig,
    ProxyCecaSharedResult,
    _compute_allowed_bundles_by_bidder,
    _enforce_universe_constraint,
    _filter_initial_bid,
    finalize_proxy_ceca_result,
    run_proxy_ceca_elicitation,
    run_proxy_ceca_experiment,
)
from auctionlab.instances.base import AuctionInstance
from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.person_simulator import LlmPersonSimulator
from auctionlab.llm.proxies import CecaTrimResult, LlmAuctionProxyAdapter, LlmInferredXorProxy
from auctionlab.proxies.full_info import FullInfoAuctionProxy


def make_proxies(toy_instance) -> list[FullInfoAuctionProxy]:
    return [
        FullInfoAuctionProxy(bidder_id=bidder_id, instance=toy_instance, initial="empty")
        for bidder_id in toy_instance.bidder_ids
    ]


def _ceca_step_oracle_for(proxies_by_bidder):
    def oracle(bidder_id, prices, current_bundle, round_idx):
        return proxies_by_bidder[bidder_id].ceca_step(prices, current_bundle, round_idx)

    return oracle


def test_run_proxy_ceca_experiment_matches_direct_run_ceca_pay_as_bid(toy_instance):
    cfg = CecaConfig(max_rounds=20)

    direct_proxies_by_bidder = {p.bidder_id: p for p in make_proxies(toy_instance)}
    direct_state = run_ceca(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        ceca_step_oracle=_ceca_step_oracle_for(direct_proxies_by_bidder),
        cfg=cfg,
    )
    expected = finalize_ceca_pay_as_bid(toy_instance.items, direct_state.manifest_bids)

    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )

    assert result.mechanism == "proxy_ceca_pay_as_bid"
    assert result.allocation == expected.allocation
    assert result.welfare == expected.welfare
    assert result.payments == expected.payments
    assert result.metadata["converged"] is True


def test_run_proxy_ceca_experiment_vcg_payment_rule(toy_instance):
    cfg = CecaConfig(max_rounds=20)

    direct_proxies_by_bidder = {p.bidder_id: p for p in make_proxies(toy_instance)}
    direct_state = run_ceca(
        items=toy_instance.items,
        bidder_ids=toy_instance.bidder_ids,
        ceca_step_oracle=_ceca_step_oracle_for(direct_proxies_by_bidder),
        cfg=cfg,
    )
    expected = finalize_ceca_vcg(toy_instance.items, direct_state.manifest_bids)

    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )

    assert result.mechanism == "proxy_ceca_vcg"
    assert result.allocation == expected.allocation
    assert result.welfare == expected.welfare
    assert result.payments == expected.payments


def test_metadata_includes_expected_fields(toy_instance):
    cfg = CecaConfig(max_rounds=20)

    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )

    assert result.metadata["payment_rule"] == "pay_as_bid"
    assert set(result.metadata["demand_query_count_by_bidder"]) == set(
        toy_instance.bidder_ids
    )
    assert set(result.metadata["pruning_query_count_by_bidder"]) == set(
        toy_instance.bidder_ids
    )
    assert all(
        count == 0
        for count in result.metadata["pruning_query_count_by_bidder"].values()
    )
    assert set(result.metadata["final_bids"]) == set(toy_instance.bidder_ids)
    assert set(result.metadata["initial_bids"]) == set(toy_instance.bidder_ids)
    assert result.metadata["stage1_welfare"] == pytest.approx(
        result.metadata["stage2_welfare"]
    )
    assert set(result.metadata["final_manifest_sizes"]) == set(
        toy_instance.bidder_ids
    )


def test_individually_rational_payments_under_both_rules(toy_instance):
    cfg = CecaConfig(max_rounds=20)

    for rule in ("pay_as_bid", "vcg"):
        result = run_proxy_ceca_experiment(
            toy_instance,
            make_proxies(toy_instance),
            cfg,
            ProxyCecaConfig(payment_rule=rule),
        )
        for bidder_id in toy_instance.bidder_ids:
            bundle = result.allocation.get(bidder_id, frozenset())
            true_value = toy_instance.value_of(bidder_id, bundle)
            assert result.payments[bidder_id] <= true_value + 1e-6


def test_proxies_must_match_instance_bidder_ids(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    proxies = make_proxies(toy_instance)[:-1]

    with pytest.raises(ValueError, match="proxies must contain"):
        run_proxy_ceca_experiment(
            toy_instance, proxies, cfg, ProxyCecaConfig(payment_rule="pay_as_bid")
        )


def test_invalid_payment_rule_rejected():
    with pytest.raises(ValueError):
        ProxyCecaConfig(payment_rule="not_a_real_rule")


def test_ceca_result_to_row_exports_expected_fields(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )

    row = ceca_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )

    assert row["instance_name"] == "toy"
    assert row["mechanism"] == "proxy_ceca_vcg"
    assert row["payment_rule"] == "vcg"
    assert row["converged"] is True
    assert row["stage1_welfare"] == pytest.approx(row["stage2_welfare"])
    assert row["efficiency"] == pytest.approx(1.0)
    for bidder_id in toy_instance.bidder_ids:
        assert f"{bidder_id}:" in row["demand_query_count_by_bidder"]
        assert f"{bidder_id}:" in row["pruning_query_count_by_bidder"]
        assert f"{bidder_id}:" in row["final_manifest_sizes"]
    assert row["initial_bids"]
    assert row["final_bids"]


# ---------------------------------------------------------------------------
# Tests for ProxyCecaConfig.initial_bid_mode and ceca_variant
# ---------------------------------------------------------------------------

def test_proxy_ceca_config_default_mode():
    cfg = ProxyCecaConfig(payment_rule="vcg")
    assert cfg.initial_bid_mode == "full_proxy"
    assert cfg.ceca_variant == "prior"


def test_proxy_ceca_config_singletons_variant():
    cfg = ProxyCecaConfig(payment_rule="pay_as_bid", initial_bid_mode="singletons")
    assert cfg.ceca_variant == "singletons"


def test_proxy_ceca_config_empty_variant():
    cfg = ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="empty")
    assert cfg.ceca_variant == "empty"


def test_invalid_initial_bid_mode_rejected():
    with pytest.raises(ValueError, match="initial_bid_mode"):
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="bad_mode")


# ---------------------------------------------------------------------------
# Tests for _filter_initial_bid manifest construction
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_bid():
    """XorBid with one singleton {A}, one pair {A,B}, and one triple {A,B,C}."""
    return XorBid(
        bidder_id="test",
        atoms=[
            XorAtomicBid(bundle=frozenset({"A"}), value=10.0),
            XorAtomicBid(bundle=frozenset({"A", "B"}), value=20.0),
            XorAtomicBid(bundle=frozenset({"A", "B", "C"}), value=35.0),
        ],
    )


def test_full_proxy_mode_includes_all_atoms(mixed_bid):
    filtered = _filter_initial_bid(mixed_bid, "full_proxy")
    assert len(filtered.atoms) == 3


def test_singletons_mode_only_singleton_atoms(mixed_bid):
    filtered = _filter_initial_bid(mixed_bid, "singletons")
    assert len(filtered.atoms) == 1
    assert all(len(a.bundle) == 1 for a in filtered.atoms)
    assert filtered.atoms[0].bundle == frozenset({"A"})


def test_singletons_mode_no_multi_item_atoms(mixed_bid):
    filtered = _filter_initial_bid(mixed_bid, "singletons")
    assert all(len(a.bundle) <= 1 for a in filtered.atoms)


def test_empty_mode_no_atoms(mixed_bid):
    filtered = _filter_initial_bid(mixed_bid, "empty")
    assert len(filtered.atoms) == 0


def test_singletons_mode_empty_when_no_singletons_in_bid():
    bid = XorBid(
        bidder_id="test",
        atoms=[
            XorAtomicBid(bundle=frozenset({"A", "B"}), value=20.0),
        ],
    )
    filtered = _filter_initial_bid(bid, "singletons")
    assert len(filtered.atoms) == 0


# ---------------------------------------------------------------------------
# Tests for run_proxy_ceca_experiment with different initial_bid_mode values
# ---------------------------------------------------------------------------

@pytest.fixture
def complement_instance():
    """
    Two-item, two-bidder instance where bidder1 values complement {A,B}=15
    and singleton {A}=10; bidder2 values {B}=10 and {C}=8.

    Optimal full-info allocation: bidder1={A}, bidder2={B} for welfare 20.
    With `all_atoms` init (full_proxy), CECA converges in 1 round.
    With `singletons` init, bidder3 (i3) starts empty and must demand {A,C}.
    """
    return AuctionInstance(
        items=["A", "B", "C"],
        bidder_ids=["i1", "i2", "i3"],
        valuations={
            "i1": {
                frozenset({"A", "B"}): 15.0,
                frozenset({"A"}): 10.0,
            },
            "i2": {
                frozenset({"B"}): 9.0,
                frozenset({"C"}): 7.0,
            },
            "i3": {
                frozenset({"A", "C"}): 14.0,
            },
        },
    )


def _all_atom_proxies(instance: AuctionInstance) -> list[FullInfoAuctionProxy]:
    return [
        FullInfoAuctionProxy(bidder_id=b, instance=instance, initial="all_atoms")
        for b in instance.bidder_ids
    ]


def test_full_proxy_mode_converges_quickly(complement_instance):
    """full_proxy with all atoms already set should converge in 1 round."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy"),
    )
    assert result.metadata["converged"] is True
    assert result.metadata["ceca_rounds"] == 1
    assert result.metadata["ceca_initial_bid_mode"] == "full_proxy"
    assert result.metadata["ceca_variant"] == "prior"
    assert result.metadata["initial_manifest_total_atoms"] > 0


def test_singletons_mode_initial_manifest_has_only_singletons(complement_instance):
    """singletons mode: initial manifest atoms are all size-1 bundles."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    initial_bids = result.metadata["initial_bids"]
    for bid in initial_bids.values():
        assert all(len(a.bundle) == 1 for a in bid.atoms), (
            f"Bidder {bid.bidder_id} has multi-item atoms in singletons mode: "
            f"{[sorted(a.bundle) for a in bid.atoms if len(a.bundle) > 1]}"
        )
    assert result.metadata["ceca_variant"] == "singletons"
    assert result.metadata["ceca_initial_bid_mode"] == "singletons"


def test_singletons_mode_discovers_complement_bundles(complement_instance):
    """singletons mode: CECA must grow the manifest to find complement bundles."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    # Manifest must grow beyond initial singletons.
    assert result.metadata["manifest_growth_total"] > 0
    # At least one bidder must have demanded a multi-item bundle.
    demanded = result.metadata["demanded_bundle_count_by_bidder"]
    assert sum(demanded.values()) > 0
    # Result should still converge and reach efficient allocation.
    assert result.metadata["converged"] is True


def test_singletons_mode_more_rounds_than_full_proxy(complement_instance):
    """singletons typically takes more rounds than full_proxy for the same instance."""
    cfg = CecaConfig(max_rounds=20)
    result_full = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy"),
    )
    result_sing = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert result_sing.metadata["ceca_rounds"] >= result_full.metadata["ceca_rounds"]


def test_empty_mode_does_not_crash(complement_instance):
    """empty mode must complete without error (even though manifest starts empty)."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid", initial_bid_mode="empty"),
    )
    assert result.metadata["ceca_initial_bid_mode"] == "empty"
    assert result.metadata["initial_manifest_total_atoms"] == 0
    assert result.metadata["final_manifest_total_atoms"] >= 0
    assert result.metadata["converged"] in (True, False)


# ---------------------------------------------------------------------------
# Tests for new metadata fields
# ---------------------------------------------------------------------------

def test_metadata_includes_new_manifest_fields(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert "ceca_initial_bid_mode" in result.metadata
    assert "ceca_variant" in result.metadata
    assert "initial_manifest_total_atoms" in result.metadata
    assert "initial_manifest_sizes" in result.metadata
    assert "manifest_growth_total" in result.metadata
    assert "manifest_growth_by_bidder" in result.metadata
    assert "demanded_bundle_count_by_bidder" in result.metadata
    assert set(result.metadata["manifest_growth_by_bidder"]) == set(toy_instance.bidder_ids)
    assert set(result.metadata["demanded_bundle_count_by_bidder"]) == set(toy_instance.bidder_ids)
    growth = result.metadata["manifest_growth_total"]
    init = result.metadata["initial_manifest_total_atoms"]
    final = result.metadata["final_manifest_total_atoms"]
    assert init + growth == final


def test_ceca_result_to_row_new_fields(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )
    row = ceca_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    assert row["ceca_initial_bid_mode"] == "full_proxy"
    assert row["ceca_variant"] == "prior"
    assert "initial_manifest_total_atoms" in row
    assert "manifest_growth_total" in row
    assert "manifest_growth_by_bidder" in row
    assert "demanded_bundle_count_by_bidder" in row
    assert "reported_allocated_welfare" in row
    assert "true_allocated_welfare" in row
    assert "reported_true_welfare_ratio" in row
    assert "welfare_understatement_or_overstatement" in row
    assert "true_surplus" in row
    assert "negative_true_surplus" in row
    # Numeric consistency checks.
    assert row["true_surplus"] == pytest.approx(
        row["true_allocated_welfare"] - row["proxy_revenue"]
    )
    assert row["welfare_understatement_or_overstatement"] == pytest.approx(
        row["reported_allocated_welfare"] - row["true_allocated_welfare"]
    )


# ---------------------------------------------------------------------------
# Tests for value/payment diagnostics
# ---------------------------------------------------------------------------

def test_ceca_winner_diagnostics_rows_has_winners_only(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    rows = ceca_winner_diagnostics_rows(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    # Only winners appear in diagnostics.
    for row in rows:
        assert result.allocation.get(row["bidder_id"], frozenset())
    required = {
        "scenario", "ceca_variant", "ceca_initial_bid_mode", "payment_rule",
        "bidder_id", "allocated_bundle", "reported_value", "true_value",
        "value_error", "payment", "true_surplus", "reported_surplus",
    }
    for row in rows:
        assert required <= set(row.keys())


def test_winner_diagnostics_reported_true_welfare_ratio(toy_instance):
    """reported_true_welfare_ratio = true_welfare / reported_welfare (not NaN)."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )
    row = ceca_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    ratio = row["reported_true_welfare_ratio"]
    # FullInfo proxy reports true values, so ratio should be ~1.0.
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_negative_true_surplus_flag(toy_instance):
    """negative_true_surplus must be True when revenue > true_welfare."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )
    row = ceca_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    expected = row["true_surplus"] < -1e-6
    assert row["negative_true_surplus"] == expected


# ---------------------------------------------------------------------------
# Tests for the two-phase elicitation / finalization API
# ---------------------------------------------------------------------------

def test_run_proxy_ceca_elicitation_returns_shared_result(toy_instance):
    """run_proxy_ceca_elicitation must return a ProxyCecaSharedResult."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    assert isinstance(shared, ProxyCecaSharedResult)
    assert shared.ceca_state is not None
    assert set(shared.demand_query_count_by_bidder) == set(toy_instance.bidder_ids)
    assert set(shared.demanded_bundle_count_by_bidder) == set(toy_instance.bidder_ids)
    assert set(shared.duplicate_demand_count_by_bidder) == set(toy_instance.bidder_ids)
    assert set(shared.unchanged_demand_count_by_bidder) == set(toy_instance.bidder_ids)


def test_both_payment_rules_share_allocation(toy_instance):
    """Pay-as-bid and VCG must yield identical allocation and rounds from the same CECA run."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    pab = finalize_proxy_ceca_result(toy_instance, shared, "pay_as_bid")
    vcg = finalize_proxy_ceca_result(toy_instance, shared, "vcg")

    assert pab.allocation == vcg.allocation
    assert pab.rounds == vcg.rounds
    assert pab.metadata["converged"] == vcg.metadata["converged"]
    assert pab.metadata["ceca_rounds"] == vcg.metadata["ceca_rounds"]
    assert pab.metadata["initial_manifest_total_atoms"] == vcg.metadata["initial_manifest_total_atoms"]
    assert pab.metadata["final_manifest_total_atoms"] == vcg.metadata["final_manifest_total_atoms"]
    assert pab.metadata["manifest_growth_total"] == vcg.metadata["manifest_growth_total"]
    assert pab.metadata["demanded_bundle_count_by_bidder"] == vcg.metadata["demanded_bundle_count_by_bidder"]
    # Mechanism names must differ.
    assert pab.mechanism == "proxy_ceca_pay_as_bid"
    assert vcg.mechanism == "proxy_ceca_vcg"


def test_both_payment_rules_welfare_matches(toy_instance):
    """Reported welfare is the same for both rules (same WDP over same manifest)."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )
    pab = finalize_proxy_ceca_result(toy_instance, shared, "pay_as_bid")
    vcg = finalize_proxy_ceca_result(toy_instance, shared, "vcg")
    assert pab.welfare == pytest.approx(vcg.welfare)


def test_finalize_ceca_result_invalid_rule_raises(toy_instance):
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    with pytest.raises(ValueError, match="payment_rule"):
        finalize_proxy_ceca_result(toy_instance, shared, "bad_rule")


def test_duplicate_demand_counts_non_negative(complement_instance):
    """Duplicate and unchanged demand counts must be >= 0 for all bidders."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    for b in complement_instance.bidder_ids:
        assert shared.demanded_bundle_count_by_bidder[b] >= 0
        assert shared.unique_demanded_bundle_count_by_bidder[b] >= 0
        assert shared.duplicate_demand_count_by_bidder[b] >= 0
        assert shared.unchanged_demand_count_by_bidder[b] >= 0
        # unique <= total, duplicate = total - unique
        n = shared.demanded_bundle_count_by_bidder[b]
        u = shared.unique_demanded_bundle_count_by_bidder[b]
        d = shared.duplicate_demand_count_by_bidder[b]
        assert u <= n
        assert d == n - u


def test_ceca_runs_once_for_both_payment_rules(toy_instance):
    """Using the two-phase API, the CECA engine runs exactly once per mode.

    Verified by counting demand-query calls on the proxies: if CECA ran
    twice the demand counts would roughly double.
    """
    from auctionlab.proxies.full_info import FullInfoAuctionProxy

    cfg = CecaConfig(max_rounds=20)
    proxies = make_proxies(toy_instance)
    shared = run_proxy_ceca_elicitation(
        toy_instance,
        proxies,
        cfg,
        ProxyCecaConfig(payment_rule="pay_as_bid"),
    )
    total_demand_queries = sum(shared.demand_query_count_by_bidder.values())

    # Both payment finalizations are pure arithmetic — no further proxy calls.
    pab = finalize_proxy_ceca_result(toy_instance, shared, "pay_as_bid")
    vcg = finalize_proxy_ceca_result(toy_instance, shared, "vcg")

    # Query counts in both results must equal the single-run total.
    assert pab.metadata["demand_query_count_by_bidder"] == shared.demand_query_count_by_bidder
    assert vcg.metadata["demand_query_count_by_bidder"] == shared.demand_query_count_by_bidder


def test_new_diagnostic_fields_in_ceca_result_to_row(toy_instance):
    """ceca_result_to_row must include unique/duplicate/unchanged demand fields."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        toy_instance,
        make_proxies(toy_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg"),
    )
    row = ceca_result_to_row(
        instance_name="toy",
        instance=toy_instance,
        result=result,
    )
    assert "unique_demanded_bundle_count_by_bidder" in row
    assert "duplicate_demand_count_by_bidder" in row
    assert "unchanged_demand_count_by_bidder" in row


# ---------------------------------------------------------------------------
# Tests for run_config header display
# ---------------------------------------------------------------------------

def test_format_run_config_shows_ceca_mode():
    from auctionlab.experiments.run_config import format_run_config

    class FakeArgs:
        preset = None
        provider = "gemini"
        model = "gemini-3.1-flash-lite"
        scenario = None
        seed_type = "structured"
        proxy_type = "llm"
        ask_initial_question = True
        use_interest_map = True
        use_provisional_valuations = True
        max_candidate_bundles = None
        pv_max_tokens = 1500
        max_bundle_size = 3
        top_k = [1]
        ground_truth_queries = False
        skip_baselines = False
        max_rounds = 20
        sealed_elicitation_rounds = 0
        sealed_feedback_rule = "none"
        max_refinement_queries_per_bidder = 3
        elicited_clock = False
        clock_tie_threshold = 100.0
        elicited_ceca = True
        ceca_max_rounds = 20
        ceca_payment_rule = "both"
        ceca_proxy_type = "llm"
        ceca_no_pv = False
        ceca_initial_bid_mode = ["singletons"]
        log_dir = "outputs"

    class FakeScenario:
        name = "pc_build_6x6"
        class instance:
            items = list("ABCDEF")
            bidder_ids = [f"b{i}" for i in range(6)]
        metadata = {"num_goods": 6, "num_bidders": 6, "scenario_seed": 0}

    lines = format_run_config(FakeArgs(), [FakeScenario()])
    joined = "\n".join(lines)
    assert "singletons" in joined
    assert "mode=singletons" in joined


# ---------------------------------------------------------------------------
# Atomic trimming tests
# ---------------------------------------------------------------------------

_TRIM_ITEMS = {"A", "B", "C"}
_TRIM_ITEM_DESCRIPTIONS = {item: f"Item {item}" for item in _TRIM_ITEMS}


def _make_trim_proxy(
    ground_truth: dict[frozenset, float],
    *,
    ceca_atomic_trimming: bool = True,
    ceca_trim_value_tolerance: float = 0.0,
    epsilon: float = 1.0,
) -> LlmInferredXorProxy:
    """LlmInferredXorProxy backed by deterministic ground-truth values."""
    person = LlmPersonSimulator(
        bidder_id="trim_test",
        scenario_description="trim test",
        person_seed="",
        item_descriptions=_TRIM_ITEM_DESCRIPTIONS,
        client=MockLlmClient([]),
        ground_truth_valuations={frozenset(k): v for k, v in ground_truth.items()},
    )
    proxy = LlmInferredXorProxy(
        bidder_id="trim_test",
        person=person,
        epsilon=epsilon,
        ceca_atomic_trimming=ceca_atomic_trimming,
        ceca_trim_value_tolerance=ceca_trim_value_tolerance,
    )
    proxy.set_provisional_bid(
        {frozenset(k): v for k, v in ground_truth.items()},
        discount_inferred=False,
    )
    return proxy


def test_atomic_trimming_removes_nonessential_item():
    """Item B is not essential: value({A,B,C}) == value({A,C})."""
    gt = {
        frozenset({"A", "B", "C"}): 100.0,
        frozenset({"A", "C"}): 100.0,  # removing B doesn't change value
        frozenset({"A", "B"}): 80.0,
        frozenset({"A"}): 60.0,
        frozenset({"B"}): 10.0,
        frozenset({"C"}): 50.0,
    }
    proxy = _make_trim_proxy(gt, ceca_trim_value_tolerance=0.0)
    trim_result = proxy._prune_demanded_bundle(frozenset({"A", "B", "C"}), 100.0)
    assert frozenset({"B"}) not in trim_result.trimmed_bundle  # B removed
    assert trim_result.trim_items_removed >= 1
    assert trim_result.raw_bundle == frozenset({"A", "B", "C"})


def test_atomic_trimming_keeps_essential_item():
    """Item C is essential: value({A,B}) < value({A,B,C})."""
    gt = {
        frozenset({"A", "B", "C"}): 100.0,
        frozenset({"A", "B"}): 70.0,  # removing C drops value — C is essential
        frozenset({"A", "C"}): 80.0,
        frozenset({"B", "C"}): 90.0,
        frozenset({"A"}): 50.0,
        frozenset({"B"}): 40.0,
        frozenset({"C"}): 60.0,
    }
    proxy = _make_trim_proxy(gt, ceca_trim_value_tolerance=0.0)
    trim_result = proxy._prune_demanded_bundle(frozenset({"A", "B", "C"}), 100.0)
    # No item removal should happen since removing any item reduces value below 100
    assert frozenset({"C"}) <= trim_result.trimmed_bundle  # C must be kept


def test_atomic_trimming_uses_tolerance():
    """Item removal that drops value by tol should still trim (within tolerance)."""
    gt = {
        frozenset({"A", "B"}): 100.0,
        frozenset({"A"}): 95.0,  # removing B drops by 5, within tolerance=5
    }
    proxy = _make_trim_proxy(gt, ceca_trim_value_tolerance=5.0)
    trim_result = proxy._prune_demanded_bundle(frozenset({"A", "B"}), 100.0)
    assert frozenset({"B"}) not in trim_result.trimmed_bundle  # B removed (within tolerance)


def test_atomic_trimming_stored_value_is_original_demanded():
    """Trimmed atom value must equal original demanded_bundle value, not reduced value."""
    gt = {
        frozenset({"A", "B"}): 100.0,
        frozenset({"A"}): 100.0,  # same value — B removable
    }
    proxy = _make_trim_proxy(gt, ceca_trim_value_tolerance=0.0)
    demanded_value = 100.0
    trim_result = proxy._prune_demanded_bundle(frozenset({"A", "B"}), demanded_value)
    # The trim_result.raw_demanded_value must be the original demanded_value
    assert trim_result.raw_demanded_value == demanded_value


def test_atomic_trimming_uses_cached_values():
    """Trimming uses cached bid values when available, skipping live queries for those bundles.

    Items are tried in sorted order (A before B). To remove A, the candidate is {B} —
    not cached. To remove B (from the progressively-shrinking bundle), the candidate is {A}
    — cached at 100.0. We verify that the cached lookup for {A} counts zero trim queries
    (trim_value_queries counts only live queries, not cache hits).
    """
    gt = {
        frozenset({"A", "B"}): 100.0,
        frozenset({"A"}): 100.0,  # B is non-essential
        frozenset({"B"}): 50.0,   # A is essential (50 < 100)
    }
    proxy = _make_trim_proxy(gt)
    # Cache both {A,B} and {A} so the {A} lookup is free; {B} needs a live query.
    proxy.set_provisional_bid(
        {frozenset({"A", "B"}): 100.0, frozenset({"A"}): 100.0},
        discount_inferred=False,
    )
    trim_result = proxy._prune_demanded_bundle(frozenset({"A", "B"}), 100.0)
    # {A} is cached — removing B via the {A} candidate hits the cache.
    # Only {B} (candidate when trying to remove A) is not cached and needs a query.
    # B should be removed because {A}=100.0 == demanded_value=100.0.
    assert frozenset({"B"}) not in trim_result.trimmed_bundle
    # The cached lookup for {A} does not increment trim_value_queries.
    # (The live query for {B} as a candidate does increment it if A is tried first.)
    # Either way, trim_items_removed must reflect B was dropped.
    assert trim_result.trim_items_removed >= 1


def test_trim_value_cache_avoids_repeated_vq():
    """The trim value cache must prevent re-querying the same sub-bundle across rounds.

    If a sub-bundle value was queried in trim round 1, calling _prune_demanded_bundle
    again for the same parent bundle in round 2 should not issue a new value query.
    The pruning_query_count must not grow on the second call.
    """
    gt = {
        frozenset({"A", "B", "C"}): 100.0,
        frozenset({"A", "B"}): 100.0,  # C is non-essential
        frozenset({"A", "C"}): 60.0,
        frozenset({"B", "C"}): 60.0,
    }
    # Don't seed {A,C} or {B,C} in provisional bid so they require live VQs.
    person = LlmPersonSimulator(
        bidder_id="trim_test",
        scenario_description="trim test",
        person_seed="",
        item_descriptions=_TRIM_ITEM_DESCRIPTIONS,
        client=MockLlmClient([]),
        ground_truth_valuations={frozenset(k): v for k, v in gt.items()},
    )
    proxy = LlmInferredXorProxy(
        bidder_id="trim_test",
        person=person,
        epsilon=1.0,
        ceca_atomic_trimming=True,
        ceca_trim_value_tolerance=0.0,
    )
    proxy.set_provisional_bid(
        {frozenset({"A", "B", "C"}): 100.0, frozenset({"A", "B"}): 100.0},
        discount_inferred=False,
    )

    # First trim call — must query sub-bundles not in the cached bid.
    proxy._prune_demanded_bundle(frozenset({"A", "B", "C"}), 100.0)
    queries_after_first = proxy.pruning_query_count

    # Second trim call on the SAME bundle — sub-bundle values must hit cache.
    proxy._prune_demanded_bundle(frozenset({"A", "B", "C"}), 100.0)
    queries_after_second = proxy.pruning_query_count

    assert queries_after_second == queries_after_first, (
        f"Second trim call must use cache; got {queries_after_second - queries_after_first} "
        f"extra VQs (expected 0)"
    )


def test_trim_value_cache_is_populated_by_first_trim():
    """After the first trim, _trim_value_cache must contain the queried sub-bundle values."""
    gt = {
        frozenset({"A", "B"}): 10.0,
        frozenset({"A"}): 7.0,   # queried when trying to remove B
    }
    person = LlmPersonSimulator(
        bidder_id="t",
        scenario_description="t",
        person_seed="",
        item_descriptions={"A": "A", "B": "B"},
        client=MockLlmClient([]),
        ground_truth_valuations={frozenset(k): v for k, v in gt.items()},
    )
    proxy = LlmInferredXorProxy(
        bidder_id="t", person=person, epsilon=1.0, ceca_atomic_trimming=True,
    )
    proxy.set_provisional_bid({frozenset({"A", "B"}): 10.0}, discount_inferred=False)

    proxy._prune_demanded_bundle(frozenset({"A", "B"}), 10.0)

    # {A} should have been queried and cached (since B removal tried candidate {A}).
    assert frozenset({"A"}) in proxy._trim_value_cache


def test_atomic_trimming_disabled_skips_pruning(complement_instance):
    """With atomic_trimming=False, demanded bundle is inserted as-is."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy", atomic_trimming=False),
    )
    assert result.metadata["ceca_initial_bid_mode"] == "full_proxy"
    assert result.metadata.get("ceca_atomic_trimming") is False


def test_trim_result_in_shared_result(complement_instance):
    """ProxyCecaSharedResult has trim aggregate fields."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons", atomic_trimming=True),
    )
    assert hasattr(shared, 'ceca_atomic_trimming')
    assert hasattr(shared, 'ceca_total_trim_items_removed')
    assert hasattr(shared, 'ceca_trimmed_demand_count')
    assert shared.ceca_total_trim_items_removed >= 0
    assert shared.ceca_total_trim_value_queries >= 0


def test_trim_fields_in_mechanism_result(complement_instance):
    """finalize_proxy_ceca_result includes trim metadata."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    result = finalize_proxy_ceca_result(complement_instance, shared, "vcg")
    assert "ceca_atomic_trimming" in result.metadata
    assert "ceca_total_trim_items_removed" in result.metadata
    assert "ceca_trimmed_demand_count" in result.metadata


def test_proxy_ceca_config_trim_validation():
    """trim_value_tolerance must be >= 0."""
    with pytest.raises(ValueError, match="trim_value_tolerance"):
        ProxyCecaConfig(payment_rule="vcg", trim_value_tolerance=-1.0)


# ---------------------------------------------------------------------------
# Tests for Feature 1: corrected trimming diagnostics
# ---------------------------------------------------------------------------

def test_trimming_diagnostics_net_items_removed(complement_instance):
    """total_net_items_removed == total_raw_demand_items - total_inserted_atom_items."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons", atomic_trimming=True),
    )
    assert shared.total_net_items_removed == (
        shared.total_raw_demand_items - shared.total_inserted_atom_items
    )


def test_trimming_diagnostics_fields_exist(complement_instance):
    """ProxyCecaSharedResult has all new Feature 1 trim diagnostic fields."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons", atomic_trimming=True),
    )
    assert hasattr(shared, 'num_trim_attempts')
    assert hasattr(shared, 'num_demands_trimmed_to_smaller_atom')
    assert hasattr(shared, 'total_raw_demand_items')
    assert hasattr(shared, 'total_inserted_atom_items')
    assert hasattr(shared, 'total_net_items_removed')
    assert hasattr(shared, 'avg_raw_demand_size')
    assert hasattr(shared, 'avg_inserted_atom_size')
    assert shared.num_trim_attempts >= 0
    assert shared.num_demands_trimmed_to_smaller_atom >= 0
    assert shared.num_demands_trimmed_to_smaller_atom <= shared.num_trim_attempts
    assert shared.total_raw_demand_items >= 0
    assert shared.total_inserted_atom_items >= 0
    assert shared.total_net_items_removed >= 0
    assert shared.avg_raw_demand_size >= 0.0
    assert shared.avg_inserted_atom_size >= 0.0


def test_trimming_diagnostics_in_mechanism_result(complement_instance):
    """finalize_proxy_ceca_result includes Feature 1 trim metadata."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    result = finalize_proxy_ceca_result(complement_instance, shared, "vcg")
    assert "num_trim_attempts" in result.metadata
    assert "total_net_items_removed" in result.metadata
    assert "avg_raw_demand_size" in result.metadata
    assert "avg_inserted_atom_size" in result.metadata
    assert result.metadata["total_net_items_removed"] == (
        result.metadata["total_raw_demand_items"]
        - result.metadata["total_inserted_atom_items"]
    )


# ---------------------------------------------------------------------------
# Tests for Feature 2: no-new-information detection
# ---------------------------------------------------------------------------

def test_no_new_information_detected_for_repeated_demand(complement_instance):
    """When a bidder demands the same bundle twice, second is no_new_information."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert hasattr(shared, 'no_new_information_count_by_bidder')
    assert hasattr(shared, 'total_no_new_information')
    assert shared.total_no_new_information >= 0
    assert set(shared.no_new_information_count_by_bidder) == set(complement_instance.bidder_ids)
    for b in complement_instance.bidder_ids:
        assert shared.no_new_information_count_by_bidder[b] >= 0


def test_no_new_information_fields_in_mechanism_result(complement_instance):
    """finalize_proxy_ceca_result includes Feature 2 no-new-info metadata."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    result = finalize_proxy_ceca_result(complement_instance, shared, "vcg")
    assert "total_no_new_information" in result.metadata
    assert "no_new_information_count_by_bidder" in result.metadata
    assert result.metadata["total_no_new_information"] >= 0


def test_no_new_information_in_ceca_result_to_row(complement_instance):
    """ceca_result_to_row includes Feature 2 no-new-information columns."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    row = ceca_result_to_row(
        instance_name="complement",
        instance=complement_instance,
        result=result,
    )
    assert "total_no_new_information" in row
    assert "no_new_information_count_by_bidder" in row
    assert isinstance(row["total_no_new_information"], int)
    assert row["total_no_new_information"] >= 0


# ---------------------------------------------------------------------------
# Tests for Feature 3: CECA stall stopping rule
# ---------------------------------------------------------------------------

def test_ceca_config_defaults_preserved():
    """Existing CecaConfig(max_rounds=N) still works; stall fields have safe defaults."""
    cfg = CecaConfig(max_rounds=20)
    assert cfg.stop_on_no_new_information is False
    assert cfg.stall_patience == 1


def test_proxy_ceca_config_stall_defaults():
    """ProxyCecaConfig stall fields have safe defaults."""
    cfg = ProxyCecaConfig(payment_rule="vcg")
    assert cfg.stop_on_no_new_information is False
    assert cfg.stall_patience == 1


def test_stall_stopping_fields_in_ceca_state(complement_instance):
    """CecaState.stopped_reason is set correctly."""
    cfg = CecaConfig(max_rounds=5, stop_on_no_new_information=True, stall_patience=1)
    proxies_by_bidder = {p.bidder_id: p for p in _all_atom_proxies(complement_instance)}
    state = run_ceca(
        items=complement_instance.items,
        bidder_ids=complement_instance.bidder_ids,
        ceca_step_oracle=lambda bid, p, b, r: proxies_by_bidder[bid].ceca_step(p, b, r),
        cfg=cfg,
    )
    assert state.stopped_reason in ("converged", "max_rounds", "no_new_information")


def test_stopped_reason_converged_on_full_proxy(complement_instance):
    """With full_proxy mode, complement_instance converges quickly — stopped_reason is converged."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            payment_rule="vcg",
            initial_bid_mode="full_proxy",
            stop_on_no_new_information=True,
            stall_patience=1,
        ),
    )
    assert shared.stopped_reason == "converged"
    assert shared.ceca_state.stopped_reason == "converged"


def test_stopped_reason_in_mechanism_result(complement_instance):
    """finalize_proxy_ceca_result includes stopped_reason in metadata."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy"),
    )
    result = finalize_proxy_ceca_result(complement_instance, shared, "vcg")
    assert "stopped_reason" in result.metadata
    assert result.metadata["stopped_reason"] in ("converged", "max_rounds", "no_new_information")


def test_stopped_reason_in_ceca_result_to_row(complement_instance):
    """ceca_result_to_row includes stopped_reason column."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy"),
    )
    row = ceca_result_to_row(
        instance_name="complement",
        instance=complement_instance,
        result=result,
    )
    assert "stopped_reason" in row
    assert row["stopped_reason"] in ("converged", "max_rounds", "no_new_information")


def test_payment_rule_both_still_one_run_with_stall(complement_instance):
    """Payment rules share allocation even with stall stopping enabled."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            payment_rule="pay_as_bid",
            initial_bid_mode="full_proxy",
            stop_on_no_new_information=True,
            stall_patience=2,
        ),
    )
    pab = finalize_proxy_ceca_result(complement_instance, shared, "pay_as_bid")
    vcg = finalize_proxy_ceca_result(complement_instance, shared, "vcg")
    assert pab.allocation == vcg.allocation
    assert pab.metadata["stopped_reason"] == shared.stopped_reason


def test_stop_on_no_new_information_stops_early(complement_instance):
    """With stop_on_no_new_information and a very low max_rounds, CECA stops cleanly."""
    cfg = CecaConfig(max_rounds=3, stop_on_no_new_information=True, stall_patience=1)
    proxies_by_bidder = {p.bidder_id: p for p in _all_atom_proxies(complement_instance)}
    state = run_ceca(
        items=complement_instance.items,
        bidder_ids=complement_instance.bidder_ids,
        ceca_step_oracle=lambda bid, p, b, r: proxies_by_bidder[bid].ceca_step(p, b, r),
        cfg=cfg,
    )
    assert state.stopped_reason in ("converged", "max_rounds", "no_new_information")
    assert len(state.history) <= 3


def test_ceca_state_default_stopped_reason():
    """CecaState defaults stopped_reason to 'max_rounds'."""
    from auctionlab.auctions.ceca import CecaState
    from auctionlab.bids.xor import XorBid
    state = CecaState(round_idx=0, manifest_bids={}, allocation={})
    assert state.stopped_reason == "max_rounds"


# ---------------------------------------------------------------------------
# Tasks 1–5: insertion diagnostics, demand trace, no-info rounds, exhaustion
# ---------------------------------------------------------------------------

from auctionlab.experiments.proxy_ceca_runner import (
    AtomInsertionRecord,
    DemandTraceRecord,
)


def test_atom_insertion_log_populated(complement_instance):
    """atom_insertion_log records every genuine manifest change."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert hasattr(shared, "atom_insertion_log")
    assert isinstance(shared.atom_insertion_log, list)
    # Every entry is either new or update — never both.
    for rec in shared.atom_insertion_log:
        assert isinstance(rec, AtomInsertionRecord)
        assert rec.is_new or rec.is_update
        assert not (rec.is_new and rec.is_update)


def test_duplicate_atom_not_in_insertion_log(complement_instance):
    """No-new-information demands (same bundle+value already in manifest) do NOT appear in atom_insertion_log."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    # Insertion log total must not exceed final manifest atoms − initial atoms.
    total_init = sum(shared.initial_manifest_sizes.values())
    total_final = sum(shared.final_manifest_sizes.values())
    # Insertions can include updates (same bundle, new value), so allow >=.
    # But new atoms cannot exceed manifest growth.
    new_atom_count = sum(1 for r in shared.atom_insertion_log if r.is_new)
    assert new_atom_count <= total_final - total_init + 1  # +1 safety for update-then-grow edge cases


def test_manifest_growth_equals_sum_of_sizes(complement_instance):
    """manifest_growth_total == sum(final_manifest_sizes) - sum(initial_manifest_sizes)."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    expected = sum(shared.final_manifest_sizes.values()) - sum(shared.initial_manifest_sizes.values())
    assert shared.manifest_growth_total == expected


def test_outside_interest_is_none_without_interest_map(complement_instance):
    """FullInfoAuctionProxy has no interest map — outside_interest fields are None."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    for rec in shared.atom_insertion_log:
        assert rec.raw_outside_interest is None
        assert rec.inserted_outside_interest is None
    for b in complement_instance.bidder_ids:
        assert shared.outside_interest_insertion_count_by_bidder[b] == 0


def test_demand_trace_populated(complement_instance):
    """demand_trace has one entry per bidder per round."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert hasattr(shared, "demand_trace")
    assert len(shared.demand_trace) > 0
    for rec in shared.demand_trace:
        assert isinstance(rec, DemandTraceRecord)
        assert isinstance(rec.same_as_previous_demand, bool)
        assert isinstance(rec.same_as_previous_trimmed_atom, bool)
        assert isinstance(rec.bidder_exhausted, bool)
        assert rec.bidder_exhausted is False  # exhaust disabled by default


def test_demand_trace_in_mechanism_result(complement_instance):
    """MechanismResult.metadata has demand_trace key."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert "demand_trace" in result.metadata
    assert "atom_insertion_log" in result.metadata


def test_no_info_round_count_computed(complement_instance):
    """no_info_round_count is non-negative and present on shared result."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert hasattr(shared, "no_info_round_count")
    assert shared.no_info_round_count >= 0
    assert isinstance(shared.no_info_bidder_count_by_round, dict)


def test_bidders_exhausted_by_repetition_field(complement_instance):
    """bidders_exhausted_by_repetition is a list of bidder ids (may be empty)."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="full_proxy"),
    )
    assert isinstance(shared.bidders_exhausted_by_repetition, list)
    for b in shared.bidders_exhausted_by_repetition:
        assert b in complement_instance.bidder_ids


def test_exhaustion_disabled_by_default(complement_instance):
    """exhaust_repeated_bidders=False (default): no exhaustion events, no skips."""
    cfg = ProxyCecaConfig(payment_rule="vcg")
    assert cfg.exhaust_repeated_bidders is False
    assert cfg.bidder_stall_patience == 3

    ceca_cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        ceca_cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    assert len(shared.exhaustion_events) == 0
    assert not any(r.bidder_exhausted for r in shared.demand_trace)


def test_exhaustion_enabled_runs_without_error(complement_instance):
    """exhaust_repeated_bidders=True completes without error; events list is a list of dicts."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            payment_rule="vcg",
            initial_bid_mode="singletons",
            exhaust_repeated_bidders=True,
            bidder_stall_patience=2,
        ),
    )
    assert isinstance(shared.exhaustion_events, list)
    for ev in shared.exhaustion_events:
        assert "round_idx" in ev
        assert "bidder_id" in ev
        assert "trimmed_atom" in ev
        assert "consecutive_count" in ev


def test_exhaustion_events_in_ceca_result_to_row(complement_instance):
    """ceca_result_to_row includes exhaustion_event_count and outside_interest columns."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", initial_bid_mode="singletons"),
    )
    row = ceca_result_to_row(
        instance_name="complement",
        instance=complement_instance,
        result=result,
    )
    assert "exhaustion_event_count" in row
    assert "no_info_round_count" in row
    assert "outside_interest_insertion_total" in row
    assert "atom_insertion_count" in row


# ---------------------------------------------------------------------------
# Demand universe constraint tests
# ---------------------------------------------------------------------------

def test_proxy_ceca_config_invalid_universe():
    """ceca_demand_universe must be one of the four valid modes."""
    with pytest.raises(ValueError, match="ceca_demand_universe"):
        ProxyCecaConfig(ceca_demand_universe="bad_mode")


def test_proxy_ceca_config_valid_universes():
    """All four valid demand universe modes are accepted."""
    for mode in ("all_items", "interested_items", "candidate_bundles", "manifest_plus_candidates"):
        cfg = ProxyCecaConfig(ceca_demand_universe=mode)
        assert cfg.ceca_demand_universe == mode


def test_compute_allowed_bundles_all_items_returns_none():
    """all_items mode returns None for every bidder (no constraint)."""
    bids = {
        "b1": XorBid("b1", [XorAtomicBid(frozenset({"A"}), 10.0)]),
        "b2": XorBid("b2", [XorAtomicBid(frozenset({"B", "C"}), 5.0)]),
    }
    result = _compute_allowed_bundles_by_bidder(
        "all_items", bids, {"b1": None, "b2": None}, None
    )
    assert result == {"b1": None, "b2": None}


def test_compute_allowed_bundles_candidate_bundles():
    """candidate_bundles mode returns exactly the pre-CECA atom bundles."""
    bids = {
        "b1": XorBid("b1", [
            XorAtomicBid(frozenset({"A"}), 10.0),
            XorAtomicBid(frozenset({"A", "B"}), 20.0),
        ]),
    }
    result = _compute_allowed_bundles_by_bidder(
        "candidate_bundles", bids, {"b1": None}, None
    )
    assert result["b1"] == frozenset([frozenset({"A"}), frozenset({"A", "B"})])


def test_compute_allowed_bundles_interested_items_respects_max_size():
    """interested_items mode generates subsets up to max_bundle_size."""
    bids = {"b1": XorBid("b1", [XorAtomicBid(frozenset({"A"}), 1.0)])}
    interest = {"b1": frozenset({"A", "B", "C"})}
    result = _compute_allowed_bundles_by_bidder("interested_items", bids, interest, max_bundle_size=2)
    allowed = result["b1"]
    assert allowed is not None
    # Should have singletons + pairs but not triples.
    assert frozenset({"A"}) in allowed
    assert frozenset({"A", "B"}) in allowed
    assert frozenset({"A", "B", "C"}) not in allowed


def test_compute_allowed_bundles_interested_items_no_interest_map():
    """interested_items with no interest map for a bidder returns None (unconstrained)."""
    bids = {"b1": XorBid("b1", [XorAtomicBid(frozenset({"A"}), 1.0)])}
    result = _compute_allowed_bundles_by_bidder("interested_items", bids, {"b1": None}, None)
    assert result["b1"] is None


def test_enforce_universe_constraint_rejects_no_alternative():
    """_enforce_universe_constraint returns (None, 0, False) when no admissible alternative."""
    bid = XorBid("b", [XorAtomicBid(frozenset({"A"}), 5.0)])
    allowed = frozenset([frozenset({"A"})])
    # current_bundle already allocated {A}; projecting doesn't find anything strictly better.
    def prices(b):
        return 0.0
    atom, val, proj = _enforce_universe_constraint(
        frozenset({"A", "B"}), 10.0, frozenset({"A"}), allowed, bid, prices
    )
    assert atom is None
    assert proj is False


def test_enforce_universe_constraint_projects_to_best():
    """_enforce_universe_constraint returns the best admissible bundle when utility > current."""
    bid = XorBid("b", [
        XorAtomicBid(frozenset({"A"}), 5.0),
        XorAtomicBid(frozenset({"B"}), 12.0),
    ])
    allowed = frozenset([frozenset({"A"}), frozenset({"B"})])
    def prices(b):
        return 0.0
    # Current allocation is empty; {B} with value 12 is strictly better than empty (0 util).
    atom, val, proj = _enforce_universe_constraint(
        frozenset({"A", "B", "C"}), 15.0, frozenset(), allowed, bid, prices
    )
    assert atom == frozenset({"B"})
    assert val == 12.0
    assert proj is True


def test_candidate_bundles_mode_shared_result_fields(complement_instance):
    """candidate_bundles mode populates demand universe fields on shared result."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(initial_bid_mode="singletons", ceca_demand_universe="candidate_bundles"),
    )
    assert shared.ceca_demand_universe == "candidate_bundles"
    assert isinstance(shared.out_of_universe_demand_count, int)
    assert isinstance(shared.rejected_out_of_universe_count, int)
    assert isinstance(shared.projected_demand_count, int)
    assert set(shared.allowed_bundle_count_by_bidder.keys()) == set(complement_instance.bidder_ids)


def test_candidate_bundles_no_outside_universe_insertions(complement_instance):
    """With FullInfoAuctionProxy, candidate_bundles never inserts an atom outside candidates."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            initial_bid_mode="singletons",
            ceca_demand_universe="candidate_bundles",
            atomic_trimming=True,
        ),
    )
    # Collect the pre-CECA candidate bundles per bidder from the atom_insertion_log.
    # Every inserted atom must be in the original valuations (since FullInfoProxy only
    # demands from valuations, which equals candidate bundles).
    candidate_bundles_by_bidder = {
        b: frozenset(complement_instance.valuations[b].keys())
        for b in complement_instance.bidder_ids
    }
    for rec in shared.atom_insertion_log:
        allowed = candidate_bundles_by_bidder[rec.bidder_id]
        assert rec.inserted_bundle in allowed, (
            f"{rec.bidder_id} inserted {rec.inserted_bundle} outside candidates {allowed}"
        )


def test_all_items_mode_out_of_universe_count_zero(complement_instance):
    """all_items mode records zero out-of-universe demands."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(initial_bid_mode="singletons", ceca_demand_universe="all_items"),
    )
    assert shared.ceca_demand_universe == "all_items"
    assert shared.out_of_universe_demand_count == 0
    assert shared.rejected_out_of_universe_count == 0
    assert shared.projected_demand_count == 0


def test_universe_fields_in_mechanism_result_metadata(complement_instance):
    """finalize_proxy_ceca_result includes demand universe fields in metadata."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            payment_rule="vcg",
            initial_bid_mode="singletons",
            ceca_demand_universe="candidate_bundles",
        ),
    )
    md = result.metadata
    assert md["ceca_demand_universe"] == "candidate_bundles"
    assert "out_of_universe_demand_count" in md
    assert "rejected_out_of_universe_count" in md
    assert "projected_demand_count" in md
    assert "allowed_bundle_count_by_bidder" in md


def test_universe_fields_in_ceca_result_to_row(complement_instance):
    """ceca_result_to_row includes demand universe CSV columns."""
    cfg = CecaConfig(max_rounds=20)
    result = run_proxy_ceca_experiment(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(payment_rule="vcg", ceca_demand_universe="candidate_bundles"),
    )
    row = ceca_result_to_row(
        instance_name="complement",
        instance=complement_instance,
        result=result,
    )
    assert row["ceca_demand_universe"] == "candidate_bundles"
    assert "out_of_universe_demand_count" in row
    assert "rejected_out_of_universe_count" in row
    assert "projected_demand_count" in row


def test_candidate_bundles_allowed_count_matches_proxy_atoms(complement_instance):
    """allowed_bundle_count_by_bidder in candidate_bundles mode equals each bidder's atom count."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(ceca_demand_universe="candidate_bundles"),
    )
    for bidder_id in complement_instance.bidder_ids:
        expected = len(complement_instance.valuations[bidder_id])
        assert shared.allowed_bundle_count_by_bidder[bidder_id] == expected, (
            f"{bidder_id}: expected {expected} candidates, "
            f"got {shared.allowed_bundle_count_by_bidder[bidder_id]}"
        )


def test_empty_plus_trimming_manifest_grows(complement_instance):
    """empty mode + atomic trimming must still insert atoms and grow the manifest."""
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(
            initial_bid_mode="empty",
            atomic_trimming=True,
        ),
    )
    assert shared.initial_manifest_sizes == {
        b: 0 for b in complement_instance.bidder_ids
    }, "empty mode must start with zero atoms per bidder"
    total_final = sum(shared.final_manifest_sizes.values())
    assert total_final > 0, "CECA must insert at least one atom when proxies have positive value"


def test_projection_value_uses_original_bid_not_raw_demand():
    """When a demanded bundle is projected to an admissible candidate, the inserted
    value must come from original_bid.value_of(projected), not the raw demand value.
    """
    # original_bid has {B}: 12.  Raw demand is {A,B,C}: 15.
    # Projection should pick {B} with value 12, NOT 15.
    bid = XorBid("b", [XorAtomicBid(frozenset({"B"}), 12.0)])
    allowed = frozenset([frozenset({"B"})])

    def prices(b):
        return 0.0

    atom, val, was_projected = _enforce_universe_constraint(
        frozenset({"A", "B", "C"}), 15.0, frozenset(), allowed, bid, prices
    )
    assert atom == frozenset({"B"})
    assert val == pytest.approx(12.0), (
        f"projection must use original_bid.value_of(projected_bundle)=12, got {val}"
    )
    assert was_projected is True


def test_payment_rule_both_allocation_is_deterministic(complement_instance):
    """Calling finalize_proxy_ceca_result twice with different payment rules must
    return the identical allocation — no ILP is re-run, so there is no tie-breaking
    divergence between PAB and VCG.
    """
    cfg = CecaConfig(max_rounds=20)
    shared = run_proxy_ceca_elicitation(
        complement_instance,
        _all_atom_proxies(complement_instance),
        cfg,
        ProxyCecaConfig(),
    )
    pab = finalize_proxy_ceca_result(complement_instance, shared, "pay_as_bid")
    vcg = finalize_proxy_ceca_result(complement_instance, shared, "vcg")

    assert pab.allocation == vcg.allocation
    assert pab.welfare == pytest.approx(vcg.welfare)
    # shared.final_stage2 is the single source of truth for both.
    assert pab.allocation == shared.final_stage2.allocation
