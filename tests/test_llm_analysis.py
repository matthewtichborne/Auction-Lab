from __future__ import annotations

import csv
from pathlib import Path

from auctionlab.experiments.llm_analysis import (
    analyze_llm_run,
    compare_llm_runs,
)
from auctionlab.experiments.llm_analysis_io import write_csv_rows
from auctionlab.instances.base import AuctionInstance
from auctionlab.instances.nl_scenarios import NaturalLanguageAuctionScenario


# Local fixture scenario, independent of the curated registry, so these
# analysis-pipeline tests don't depend on which scenarios happen to be
# curated elsewhere.
ELECTRONICS_COMPLEMENTS_SCENARIO = NaturalLanguageAuctionScenario(
    name="electronics_complements",
    seed_type="explicit",
    instance=AuctionInstance(
        items=["IPAD", "PENCIL"],
        bidder_ids=["artist", "listener"],
        valuations={
            "artist": {
                frozenset({"IPAD"}): 500.0,
                frozenset({"PENCIL"}): 120.0,
                frozenset({"IPAD", "PENCIL"}): 650.0,
            },
            "listener": {
                frozenset({"IPAD"}): 100.0,
                frozenset({"PENCIL"}): 20.0,
                frozenset({"IPAD", "PENCIL"}): 110.0,
            },
        },
    ),
    scenario_description=(
        "A small auction of electronics. The items are an iPad and "
        "an Apple Pencil."
    ),
    item_descriptions={
        "IPAD": "Apple iPad tablet suitable for drawing, note-taking, and digital work.",
        "PENCIL": "Apple Pencil stylus for digital art and precise handwriting.",
    },
    person_seeds={
        "artist": (
            "The artist is mainly interested in digital art. They value "
            "the Apple Pencil at about $120, the iPad at about $500, and "
            "the combination at about $650."
        ),
        "listener": (
            "The listener has only mild interest in these items. They "
            "value the iPad at about $100, the Pencil at about $20, and "
            "the combination at about $110."
        ),
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def sealed_result_row() -> dict[str, str]:
    return {
        "instance_name": "electronics_complements",
        "efficiency": "1.0",
        "llm_proxy_true_welfare": "670.0",
        "llm_proxy_reported_welfare": "660.0",
        "full_info_welfare": "670.0",
        "llm_proxy_query_count": "6",
        "allocation_match": "True",
        "welfare_match": "True",
        "llm_proxy_reported_bids": (
            "artist={[IPAD]:500.0;[PENCIL]:100.0}|"
            "listener={[PENCIL]:20.0}"
        ),
        "full_info_allocation": "artist:[IPAD];listener:[PENCIL]",
        "llm_proxy_allocation": "artist:[IPAD];listener:[PENCIL]",
    }


def clock_result_row() -> dict[str, str]:
    return {
        "instance_name": "electronics_complements",
        "efficiency": "0.9701492537313433",
        "clock_llm_true_welfare": "650.0",
        "clock_llm_reported_welfare": "640.0",
        "full_info_welfare": "670.0",
        "clock_llm_query_count": "10",
        "clock_rounds": "4",
        "allocation_match": "False",
        "welfare_match": "False",
        "clock_llm_reported_bids": (
            "artist={[IPAD,PENCIL]:640.0}|listener={}"
        ),
        "full_info_allocation": "artist:[IPAD];listener:[PENCIL]",
        "clock_llm_allocation": "artist:[IPAD,PENCIL];listener:[]",
    }


def proxy_sealed_result_row() -> dict[str, str]:
    return {
        "instance_name": "electronics_complements",
        "mechanism": "proxy_sealed_vcg_elicited_all_provisional_1",
        "full_info_welfare": "670.0",
        "proxy_reported_welfare": "630.0",
        "proxy_true_welfare": "630.0",
        "efficiency": "0.9402985074626866",
        "full_info_revenue": "0.0",
        "proxy_revenue": "0.0",
        "allocation_match": "False",
        "welfare_match": "False",
        "full_info_allocation": "artist:[IPAD];listener:[PENCIL]",
        "proxy_allocation": "artist:[IPAD,PENCIL];listener:[]",
        "elicitation_rounds": "1",
        "feedback_rule": "all_provisional",
        "max_refinements_per_bidder": "2",
        "refinement_query_count_by_bidder": "artist:1;listener:1",
        "initial_bids": (
            "artist={[IPAD]:500.0;[PENCIL]:100.0}|listener={[PENCIL]:20.0}"
        ),
        "final_bids": "artist={[IPAD,PENCIL]:630.0}|listener={}",
    }


def proxy_clock_result_row() -> dict[str, str]:
    return {
        "instance_name": "electronics_complements",
        "mechanism": "proxy_clock_vcg_elicited_top_1",
        "full_info_welfare": "670.0",
        "proxy_reported_welfare": "620.0",
        "proxy_true_welfare": "620.0",
        "efficiency": "0.9253731343283582",
        "full_info_revenue": "0.0",
        "proxy_revenue": "0.0",
        "rounds": "5",
        "allocation_match": "False",
        "welfare_match": "False",
        "full_info_allocation": "artist:[IPAD];listener:[PENCIL]",
        "proxy_allocation": "artist:[IPAD,PENCIL];listener:[]",
        "top_k": "1",
        "elicited": "True",
        "margin_threshold": "100.0",
        "tie_threshold": "100.0",
        "max_refinements_per_bidder": "2",
        "refinement_query_count_by_bidder": "artist:2;listener:2",
        "final_bids": "artist={[IPAD,PENCIL]:620.0}|listener={}",
        "final_prices": "{'IPAD': 0.0, 'PENCIL': 0.0}",
    }


def summary_row() -> dict[str, str]:
    return {
        "scenario": "electronics_complements",
        "seed_type": "explicit",
        "mechanism": "sealed_llm_proxy_vcg",
        "top_k": "",
        "efficiency": "1.0",
        "true_welfare": "670.0",
        "reported_welfare": "660.0",
        "full_info_welfare": "670.0",
        "query_count": "6",
        "rounds": "",
        "allocation_match": "True",
        "welfare_match": "True",
        "inferred_bids": "",
    }


def test_analyze_llm_run_writes_all_outputs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "analysis"
    write_csv_rows(
        input_dir / "curated_sealed_llm_comparison.csv",
        list(sealed_result_row()),
        [sealed_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_clock_llm_comparison_top_2.csv",
        list(clock_result_row()),
        [clock_result_row()],
    )

    counts = analyze_llm_run(
        input_dir,
        output_dir,
        scenarios=[ELECTRONICS_COMPLEMENTS_SCENARIO],
    )

    expected_files = {
        "curated_summary.csv",
        "curated_summary_aggregate.csv",
        "llm_value_errors.csv",
        "llm_value_error_aggregate.csv",
        "llm_allocation_losses.csv",
        "llm_allocation_loss_aggregate.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert counts["static_sealed_rows"] == 1
    assert counts["static_clock_rows"] == 1
    assert counts["proxy_sealed_rows"] == 0
    assert counts["proxy_clock_rows"] == 0
    assert counts["summary_rows"] == 2
    assert counts["value_error_records"] == 4
    assert counts["allocation_loss_records"] == 4

    summary_rows = read_rows(output_dir / "curated_summary.csv")
    assert [row["mechanism"] for row in summary_rows] == [
        "sealed_llm_proxy_vcg",
        "clock_llm_proxy_vcg",
    ]
    assert summary_rows[1]["top_k"] == "2"


def test_analyze_llm_run_includes_proxy_mediated_rows(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "analysis"
    write_csv_rows(
        input_dir / "curated_sealed_llm_comparison.csv",
        list(sealed_result_row()),
        [sealed_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_clock_llm_comparison_top_2.csv",
        list(clock_result_row()),
        [clock_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_sealed_proxy_elicited.csv",
        list(proxy_sealed_result_row()),
        [proxy_sealed_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_clock_proxy_elicited_top_1.csv",
        list(proxy_clock_result_row()),
        [proxy_clock_result_row()],
    )

    counts = analyze_llm_run(
        input_dir,
        output_dir,
        scenarios=[ELECTRONICS_COMPLEMENTS_SCENARIO],
    )

    assert counts["static_sealed_rows"] == 1
    assert counts["static_clock_rows"] == 1
    assert counts["proxy_sealed_rows"] == 1
    assert counts["proxy_clock_rows"] == 1
    assert counts["summary_rows"] == 4
    # 4 static records (as above) plus 1 atom each for the two proxy rows.
    assert counts["value_error_records"] == 6
    # 4 static records (2 bidders x 2 rows) plus 2 bidders x 2 proxy rows.
    assert counts["allocation_loss_records"] == 8

    summary_rows = read_rows(output_dir / "curated_summary.csv")
    summary_mechanisms = {row["mechanism"] for row in summary_rows}
    assert "proxy_sealed_vcg_elicited_all_provisional_1" in summary_mechanisms
    assert "proxy_clock_vcg_elicited_top_1" in summary_mechanisms

    proxy_clock_summary = next(
        row
        for row in summary_rows
        if row["mechanism"] == "proxy_clock_vcg_elicited_top_1"
    )
    assert proxy_clock_summary["top_k"] == "1"
    assert proxy_clock_summary["rounds"] == "5"
    assert proxy_clock_summary["inferred_bids"] == (
        "artist={[IPAD,PENCIL]:620.0}|listener={}"
    )

    value_error_rows = read_rows(output_dir / "llm_value_errors.csv")
    value_error_mechanisms = {row["mechanism"] for row in value_error_rows}
    assert (
        "proxy_sealed_vcg_elicited_all_provisional_1"
        in value_error_mechanisms
    )
    assert "proxy_clock_vcg_elicited_top_1" in value_error_mechanisms

    allocation_loss_rows = read_rows(
        output_dir / "llm_allocation_losses.csv"
    )
    allocation_loss_mechanisms = {
        row["mechanism"] for row in allocation_loss_rows
    }
    assert (
        "proxy_sealed_vcg_elicited_all_provisional_1"
        in allocation_loss_mechanisms
    )
    assert "proxy_clock_vcg_elicited_top_1" in allocation_loss_mechanisms

    # The proxy sealed allocation differs from full-info, so both bidders'
    # allocations changed -> welfare loss should be > 0.
    proxy_sealed_loss_rows = [
        row
        for row in allocation_loss_rows
        if row["mechanism"] == "proxy_sealed_vcg_elicited_all_provisional_1"
    ]
    assert len(proxy_sealed_loss_rows) == 2
    assert any(row["changed"] == "True" for row in proxy_sealed_loss_rows)


def test_compare_llm_runs_includes_proxy_mechanisms(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "analysis"
    write_csv_rows(
        input_dir / "curated_sealed_llm_comparison.csv",
        list(sealed_result_row()),
        [sealed_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_clock_llm_comparison_top_2.csv",
        list(clock_result_row()),
        [clock_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_sealed_proxy_elicited.csv",
        list(proxy_sealed_result_row()),
        [proxy_sealed_result_row()],
    )
    write_csv_rows(
        input_dir / "curated_clock_proxy_elicited_top_1.csv",
        list(proxy_clock_result_row()),
        [proxy_clock_result_row()],
    )

    analyze_llm_run(
        input_dir,
        output_dir,
        scenarios=[ELECTRONICS_COMPLEMENTS_SCENARIO],
    )

    rows = compare_llm_runs([("run", output_dir)])
    mechanisms = {row["mechanism"] for row in rows}

    assert "proxy_sealed_vcg_elicited_all_provisional_1" in mechanisms
    assert "proxy_clock_vcg_elicited_top_1" in mechanisms

    proxy_sealed_row = next(
        row
        for row in rows
        if row["mechanism"] == "proxy_sealed_vcg_elicited_all_provisional_1"
    )
    assert proxy_sealed_row["value_mae"] != ""
    assert proxy_sealed_row["welfare_loss"] != ""


def test_compare_llm_runs_joins_available_diagnostics(tmp_path):
    run_dir = tmp_path / "analyzed"
    write_csv_rows(
        run_dir / "curated_summary.csv",
        list(summary_row()),
        [summary_row()],
    )
    value_row = {
        "scenario": "electronics_complements",
        "seed_type": "explicit",
        "mechanism": "sealed_llm_proxy_vcg",
        "top_k": "",
        "mae": "5.0",
    }
    allocation_row = {
        "scenario": "electronics_complements",
        "seed_type": "explicit",
        "mechanism": "sealed_llm_proxy_vcg",
        "top_k": "",
        "changed_bidder_count": "1",
        "welfare_loss": "20.0",
    }
    write_csv_rows(
        run_dir / "llm_value_error_aggregate.csv",
        list(value_row),
        [value_row],
    )
    write_csv_rows(
        run_dir / "llm_allocation_loss_aggregate.csv",
        list(allocation_row),
        [allocation_row],
    )

    row = compare_llm_runs([("hosted", run_dir)])[0]

    assert row["run_label"] == "hosted"
    assert row["value_mae"] == "5.0"
    assert row["changed_bidder_count"] == "1"
    assert row["welfare_loss"] == "20.0"


def test_compare_llm_runs_leaves_missing_diagnostics_blank(tmp_path):
    run_dir = tmp_path / "summary_only"
    write_csv_rows(
        run_dir / "curated_summary.csv",
        list(summary_row()),
        [summary_row()],
    )

    row = compare_llm_runs([("local", run_dir)])[0]

    assert row["value_mae"] == ""
    assert row["value_rmse"] == ""
    assert row["changed_bidder_count"] == ""
    assert row["welfare_loss"] == ""
