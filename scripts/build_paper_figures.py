"""Additive paper-only figures built from the frozen analytic package.

Reads the tracked CSV tables in ``outputs/final/analytic_package_v3/tables``
and writes new figures to ``outputs/paper_figures``. It never writes into the
frozen analytic package, so the tracked result artifacts are untouched.

Two figures are produced:

``efficiency_scaling_seeds``
    The scaling panels with the five seed-level observations drawn behind each
    cell mean. The analytic package's own version plots means only, so this is
    the figure whose caption can honestly claim to show per-seed dispersion.

``payment_vs_efficiency``
    Allocation efficiency against normalised payment error for all 95 paired
    cases, which is the paper's central allocation-versus-payment claim.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLES = Path("outputs/final/analytic_package_v3/tables")
OUTPUT = Path("outputs/paper_figures")

SERIES = ("bidders", "goods", "joint")
SERIES_LABELS = {
    "bidders": "8 goods; bidders vary",
    "goods": "8 bidders; goods vary",
    "joint": "Goods and bidders vary",
}
# Must match scripts/build_final_analytic_package.py so that colour meaning is
# stable across every figure in the paper (Okabe-Ito, colourblind-safe).
COLORS = {"Initial PV": "#777777", "Sealed": "#0072B2", "Clock": "#D55E00"}
MARKERS = {"Initial PV": "o", "Sealed": "s", "Clock": "^"}


def read(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open() as handle:
        return list(csv.DictReader(handle))


def style(ax) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def expand_anchor(row: dict[str, str]):
    """The 8x8 anchor is stored once but belongs to all three visual paths."""
    if row["series"] != "anchor":
        yield row
        return
    for series in SERIES:
        copy = dict(row)
        copy["series"] = series
        copy["x_value"] = "8"
        yield copy


def scaling_with_seeds(
    cases: list[dict[str, str]],
    mechanism_keys: tuple[tuple[str, str], ...],
    ylabel: str,
    name: str,
    *,
    scale: float = 100.0,
    legend_loc: str = "lower left",
) -> None:
    """Scaling panels with the five seed observations drawn behind each mean."""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    for ax, series in zip(axes, SERIES):
        rows = [r for r in cases if r["series"] == series]
        xs = sorted({int(r["x_value"]) for r in rows})
        for mechanism, key in mechanism_keys:
            for x in xs:
                ys = [
                    scale * float(r[key]) for r in rows if int(r["x_value"]) == x
                ]
                ax.plot(
                    [x] * len(ys), ys, marker=".", linestyle="none",
                    color=COLORS[mechanism], alpha=0.30, markersize=5,
                )
            means = [
                mean(scale * float(r[key]) for r in rows if int(r["x_value"]) == x)
                for x in xs
            ]
            ax.plot(
                xs, means, marker=MARKERS[mechanism], color=COLORS[mechanism],
                label=mechanism, linewidth=1.8, markersize=5,
            )
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Auction size")
        ax.set_xticks(range(4, 11))
        style(ax)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, loc=legend_loc)
    save(fig, name)


def payment_vs_efficiency(cases: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for mechanism, eff_key, pay_key in (
        ("Sealed", "sealed_efficiency", "sealed_payment_error_over_optimum_welfare"),
        ("Clock", "clock_efficiency", "clock_payment_error_over_optimum_welfare"),
    ):
        xs = [100 * float(r[eff_key]) for r in cases]
        ys = [100 * float(r[pay_key]) for r in cases]
        ax.scatter(
            xs, ys, s=26, alpha=0.55, label=mechanism,
            color=COLORS[mechanism], edgecolor="none",
        )
    ax.axvline(100, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Allocation efficiency (%)")
    ax.set_ylabel(r"Payment error / optimum welfare (%)")
    ax.legend(frameon=False)
    style(ax)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    save(fig, "payment_vs_efficiency")


def main() -> None:
    cases: list[dict[str, str]] = []
    for row in read("paired_case_metrics.csv"):
        cases.extend(expand_anchor(row))

    scaling_with_seeds(
        cases,
        (
            ("Initial PV", "initial_efficiency"),
            ("Sealed", "sealed_efficiency"),
            ("Clock", "clock_efficiency"),
        ),
        "Efficiency (%)",
        "efficiency_scaling_seeds",
    )
    scaling_with_seeds(
        cases,
        (
            ("Sealed", "sealed_value_queries"),
            ("Clock", "clock_value_queries"),
        ),
        "Exact value queries",
        "value_queries_scaling_seeds",
        scale=1.0,
        legend_loc="upper left",
    )
    scaling_with_seeds(
        cases,
        (
            ("Sealed", "sealed_payment_error_over_optimum_welfare"),
            ("Clock", "clock_payment_error_over_optimum_welfare"),
        ),
        "Payment error / optimum welfare (%)",
        "payment_error_scaling_seeds",
        legend_loc="upper left",
    )

    # the scatter must not double-count the 8x8 anchor
    unique = read("paired_case_metrics.csv")
    payment_vs_efficiency(unique)
    for name in (
        "efficiency_scaling_seeds",
        "value_queries_scaling_seeds",
        "payment_error_scaling_seeds",
        "payment_vs_efficiency",
    ):
        print(f"wrote {OUTPUT}/{name}.{{png,pdf}}")
    print(f"cases in scatter: {len(unique)}")


if __name__ == "__main__":
    main()
