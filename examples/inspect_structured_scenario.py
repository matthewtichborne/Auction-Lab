"""Diagnostic inspector for generated PC-build structured scenarios.

Usage::

    ./venv/bin/python examples/inspect_structured_scenario.py \
        --num-goods 6 --num-bidders 6 --scenario-seed 0 --top-k 10
"""

from __future__ import annotations

import argparse
import textwrap
from collections import Counter

from auctionlab.instances.structured import (
    _ARCHETYPE_ORDER,
    make_pc_build_scenario,
)
from auctionlab.auctions.sealed_vcg import run_sealed_xor_vcg


_BAR = "─" * 60
_WIDE = "═" * 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a generated PC-build structured scenario."
    )
    p.add_argument("--num-goods", type=int, default=6)
    p.add_argument("--num-bidders", type=int, default=6)
    p.add_argument("--scenario-seed", type=int, default=0)
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top bundles by true value to print per bidder.",
    )
    return p.parse_args()


def _wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, subsequent_indent=indent)


def _bundle_str(bundle: frozenset) -> str:
    return "{" + ", ".join(sorted(bundle)) + "}"


def main() -> None:
    args = parse_args()

    scenario = make_pc_build_scenario(
        num_goods=args.num_goods,
        num_bidders=args.num_bidders,
        seed=args.scenario_seed,
    )
    instance = scenario.instance

    # ------------------------------------------------------------------
    # 1. Header — scenario overview
    # ------------------------------------------------------------------
    print()
    print(_WIDE)
    print(f"  Scenario   {scenario.name}")
    print(f"  Seed type  {scenario.seed_type}")
    print(_WIDE)

    md = scenario.metadata
    print(f"  num_goods               {md['num_goods']}")
    print(f"  num_bidders             {md['num_bidders']}")
    print(f"  scenario_seed           {md['scenario_seed']}")
    print(f"  full_valuation_table_size {md['full_valuation_table_size']}")
    print(f"  domain                  {md.get('domain', '—')}")
    print(f"  valuation_model         {md.get('valuation_model', '—')}")
    print(f"  seed_style              {md.get('seed_style', '—')}")

    print()
    print(f"  Items ({len(instance.items)}): {', '.join(instance.items)}")
    print(f"  Bidders ({len(instance.bidder_ids)}): {', '.join(instance.bidder_ids)}")

    # Scenario description
    print()
    print(_BAR)
    print("  Scenario description")
    print(_BAR)
    print(_wrap(scenario.scenario_description))

    # ------------------------------------------------------------------
    # 2. Person seeds
    # ------------------------------------------------------------------
    print()
    print(_WIDE)
    print("  Person seeds")
    print(_WIDE)
    for bidder_id in instance.bidder_ids:
        seed = scenario.person_seeds[bidder_id]
        print()
        print(f"  ── {bidder_id} {'─' * max(0, 54 - len(bidder_id))}")
        for paragraph in seed.split("\n\n"):
            print(_wrap(paragraph.strip()))
            print()

    # ------------------------------------------------------------------
    # 3. Per-bidder valuation diagnostics
    # ------------------------------------------------------------------
    print(_WIDE)
    print("  Valuation diagnostics")
    print(_WIDE)

    for bidder_id in instance.bidder_ids:
        table = instance.valuations[bidder_id]
        values = list(table.values())
        nonzero_values = [v for v in values if v > 0]

        unique_levels = len(set(round(v, 2) for v in values))
        zero_count = len(values) - len(nonzero_values)
        max_val = max(values)
        min_nonzero = min(nonzero_values) if nonzero_values else 0.0

        # Cap hit rate: use profile metadata if available, else infer from distribution
        profile_md = md.get("profiles", {}).get(bidder_id, {})
        cap_val = profile_md.get("budget_cap")
        top_val_count = sum(1 for v in values if abs(v - max_val) < 0.01) if max_val > 0 else 0
        if cap_val is not None:
            cap_hits = sum(1 for v in values if abs(v - cap_val) < 0.01)
            cap_note = (
                f"  (budget_cap={cap_val:.0f}, {cap_hits}/{len(values)} bundles at cap"
                f" = {cap_hits/len(values):.1%})"
            )
        else:
            pct = top_val_count / len(values) if values else 0.0
            cap_note = (
                f"  (largest value {max_val:.0f} shared by {top_val_count} bundles"
                + (" — large plateau, check saturation" if pct >= 0.25 else "")
                + ")"
            )

        print()
        print(f"  ── {bidder_id} {'─' * max(0, 54 - len(bidder_id))}")
        print(f"  bundles:        {len(table)}")
        print(f"  unique levels:  {unique_levels}")
        print(f"  value range:    {min_nonzero:.0f} – {max_val:.0f}")
        print(f"  zero-value:     {zero_count}")
        print(cap_note)

        # Top-k bundles by true value
        sorted_bundles = sorted(table.items(), key=lambda kv: -kv[1])
        print(f"\n  Top {args.top_k} bundles by true value:")
        print(f"    {'bundle':<40}  {'value':>8}")
        print(f"    {'─'*40}  {'─'*8}")
        for bundle, val in sorted_bundles[: args.top_k]:
            b_str = _bundle_str(bundle)
            print(f"    {b_str:<40}  {val:>8.0f}")

    # ------------------------------------------------------------------
    # 4. Full-information sealed VCG allocation
    # ------------------------------------------------------------------
    print()
    print(_WIDE)
    print("  Full-information sealed VCG allocation")
    print(_WIDE)

    bids = instance.to_xor_bids()
    result = run_sealed_xor_vcg(items=instance.items, bids=bids)

    print(f"\n  Reported welfare: {result.welfare:.0f}")
    print(f"  Revenue:          {sum(result.payments.values()):.0f}")
    print()
    print(f"  {'bidder':<22}  {'allocation':<36}  {'payment':>8}  {'value':>8}")
    print(f"  {'─'*22}  {'─'*36}  {'─'*8}  {'─'*8}")

    total_true_welfare = 0.0
    for bidder_id in instance.bidder_ids:
        alloc = result.allocation.get(bidder_id, frozenset())
        payment = result.payments.get(bidder_id, 0.0)
        true_val = instance.value_of(bidder_id, alloc)
        total_true_welfare += true_val
        alloc_str = _bundle_str(alloc) if alloc else "∅"
        print(
            f"  {bidder_id:<22}  {alloc_str:<36}  {payment:>8.0f}  {true_val:>8.0f}"
        )

    print(f"\n  True welfare (ground truth): {total_true_welfare:.0f}")

    # Efficiency
    all_bundles = [
        (bidder_id, b, v)
        for bidder_id in instance.bidder_ids
        for b, v in instance.valuations[bidder_id].items()
    ]
    # Upper-bound true welfare via WDP on ground-truth valuations
    from auctionlab.solvers.wdp_ilp import solve_wdp_xor_ilp
    true_wdp = solve_wdp_xor_ilp(instance.items, bids)
    eff = total_true_welfare / true_wdp.welfare if true_wdp.welfare > 0 else float("nan")
    print(f"  Full-info WDP welfare:       {true_wdp.welfare:.0f}")
    print(f"  Efficiency:                  {eff:.1%}")

    # ------------------------------------------------------------------
    # 5. Allocation concentration
    # ------------------------------------------------------------------
    print()
    print(_BAR)
    print("  Allocation concentration")
    print(_BAR)

    all_allocations = {
        bidder_id: result.allocation.get(bidder_id, frozenset())
        for bidder_id in instance.bidder_ids
    }
    non_empty_winners = sum(1 for alloc in all_allocations.values() if alloc)
    allocated_goods = frozenset().union(*all_allocations.values())
    largest_bundle_size = max((len(alloc) for alloc in all_allocations.values()), default=0)
    winner_good_share = len(allocated_goods) / len(instance.items) if instance.items else 0.0

    print(f"  non_empty_winners:             {non_empty_winners}")
    print(f"  largest_allocated_bundle_size: {largest_bundle_size}")
    print(f"  allocated_goods:               {len(allocated_goods)}")
    print(f"  winner_good_share:             {winner_good_share:.1%}")


if __name__ == "__main__":
    main()
