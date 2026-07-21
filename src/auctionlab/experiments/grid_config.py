"""Reusable experiment-grid configuration schema.

Distinguishes the methodological categories described in
``docs/parameter_tuning_methodology.md``:

1. **Environment** -- scenario spec / size / seed.
2. **Proxy candidate policy** -- how candidate bundles are generated and
   priced (interest map, provisional valuations, caps).
3. **Valuation calibration** -- proxy value-inference parameters that should
   be selected on a separate bundle-pricing benchmark
   (``scripts/run_value_calibration_benchmark.py``), not on final auction
   outcomes.
4. **Sealed treatment variables** -- feedback rule / elicitation rounds:
   experimental treatments to compare, not tuning knobs.
5. **Clock treatment/development variables** -- ``top_k`` is a treatment;
   tie/margin thresholds and price step are legitimate development-tuned
   settings, fixed before final evaluation.
6. **Safety limits** -- high backstops against runaway query volume, not
   tuning targets.

Each :class:`ExperimentArm` maps 1:1 onto one invocation of
``examples/run_live_llm_curated_batch.py`` via :meth:`ExperimentArm.to_cli_args`.
This module only builds and validates configuration; it never invokes an
LLM or launches a subprocess itself.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ArmLabel = Literal["value_calibration", "development", "final", "robustness"]


class EnvironmentSpec(BaseModel):
    """Category 1: which auction environment an arm runs on."""

    scenario: list[str] = Field(default_factory=lambda: ["pc_build"])
    scenario_spec: str | None = None
    seed_type: Literal["explicit", "implicit", "structured", "all"] = "structured"
    num_goods: int = 8
    num_bidders: int = 8
    scenario_seed: int = 0


class ProxyCandidatePolicy(BaseModel):
    """Category 2: candidate-bundle generation and pricing policy.

    ``max_candidate_bundles`` and ``max_bundle_size`` are left unset (``None``)
    by default -- an explicit cap is a deliberate robustness/ablation choice,
    never an implicit default for a main/final experiment (see
    ``docs/parameter_tuning_methodology.md``, decision 2 and 4).
    """

    proxy_type: Literal["llm", "dnf", "hybrid"] = "llm"
    ask_initial_question: bool = False
    use_interest_map: bool = False
    use_provisional_valuations: bool = False
    pv_max_tokens: int = 1500
    max_candidate_bundles: int | None = None
    max_bundle_size: int | None = None
    # Reserved for a future named candidate-selection policy over the
    # interest map (e.g. "complements_first"); unused by the current CLI.
    interest_map_candidate_policy: str | None = None
    refinement_strategy: Literal["value_query", "demand_query"] = "value_query"
    hybrid_alpha: int = 10
    hybrid_delta: float = 0.95


class ValuationCalibration(BaseModel):
    """Category 3: proxy valuation-scale parameters.

    Select these on ``scripts/run_value_calibration_benchmark.py``'s
    out-of-sample bundle-pricing benchmark, then freeze them for auction
    experiments -- do not tune against final PC-build auction welfare.
    """

    epsilon: float = 1.0
    discount_inferred: bool = False
    disable_anchor_values: bool = False


class SealedTreatment(BaseModel):
    """Category 4: sealed-mechanism experimental treatment variables."""

    sealed_feedback_rule: Literal[
        "none",
        "allocated_bundle",
        "lost_interested_bundle",
        "all_provisional",
        "competitive",
        "all_valued_bundles",
    ] = "none"
    sealed_elicitation_rounds: int = 0


class ClockTreatment(BaseModel):
    """Category 5: clock experimental treatment + development-tuned settings.

    ``top_k`` is the experimental treatment (compare 1/2/3); tie/margin
    thresholds and price step are development-environment-tunable and then
    frozen for final evaluation.
    """

    top_k: list[int] = Field(default_factory=lambda: [1])
    clock_tie_threshold: float = 100.0
    clock_margin_threshold: float = 100.0
    price_step: float = 50.0
    max_rounds: int = 20
    reserve: float = 0.0


def _migrate_legacy_refinement_key(
    data: dict[str, Any], *, old_key: str, new_key: str
) -> None:
    """Map a deprecated ``old_key`` onto ``new_key`` in-place, in this
    schema's ``null``-means-unbounded terms.

    The legacy convention (still used by ``ProxySealedConfig``/
    ``ProxyClockConfig`` and the CLI) is ``0`` meaning "unlimited". This
    schema's convention is ``null``/omitted meaning "unlimited", so a legacy
    ``0`` is accepted only as a deprecated spelling of that (with a
    warning); a legacy positive value carries over unchanged. ``new_key``,
    if already present, always wins outright -- ``old_key`` is only
    consulted as a fallback.
    """
    if new_key in data or old_key not in data:
        return
    value = data.pop(old_key)
    if value == 0:
        warnings.warn(
            f"{old_key}=0 is a deprecated spelling of 'unlimited' -- use "
            f"{new_key}=null (or omit the field) instead. 0 is no longer "
            f"accepted as a literal value for {new_key}.",
            DeprecationWarning,
            stacklevel=2,
        )
        data[new_key] = None
    else:
        data[new_key] = value


class SafetyLimits(BaseModel):
    """Category 6: safety backstops, not tuning targets.

    ``null``/omitted means "no limit" -- refinement count is meant to be an
    *outcome* of the elicitation events and mechanism (feedback rule,
    elicitation rounds, clock ``top_k``), not a quantity to tune. ``0`` is
    deliberately **not** a valid explicit value for either field: it reads
    too easily as "zero refinements allowed" rather than its old meaning of
    "unlimited", so new configs must use ``null`` for that. The deprecated
    field names (``max_refinement_queries_per_bidder`` /
    ``max_total_refinement_queries``, matching the CLI flags and
    ``ProxySealedConfig``/``ProxyClockConfig``'s ``0``-means-unlimited
    convention) are still accepted on input as backwards-compatible
    aliases -- a legacy ``0`` under either old name is translated to
    ``null`` with a ``DeprecationWarning``; a legacy positive value carries
    over unchanged.

    - ``per_bidder_refinement_query_limit``: ``null`` = no per-bidder cap;
      positive int = at most that many refinement value queries per
      bidder. Only meaningful as an explicitly labelled ``"robustness"``
      arm testing a fixed query budget (e.g. ``3``) -- see
      :func:`grid_methodology_warnings`.
    - ``global_refinement_query_safety_limit``: ``null`` = no global cap;
      positive int = stop refinement once that many refinement queries
      have fired across all bidders combined. A guardrail against runaway
      query volume, not an economic tuning parameter.
    """

    per_bidder_refinement_query_limit: int | None = None
    global_refinement_query_safety_limit: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        _migrate_legacy_refinement_key(
            data,
            old_key="max_refinement_queries_per_bidder",
            new_key="per_bidder_refinement_query_limit",
        )
        _migrate_legacy_refinement_key(
            data,
            old_key="max_total_refinement_queries",
            new_key="global_refinement_query_safety_limit",
        )
        return data

    @field_validator(
        "per_bidder_refinement_query_limit", "global_refinement_query_safety_limit"
    )
    @classmethod
    def _reject_zero_and_negative(cls, value: int | None) -> int | None:
        if value == 0:
            raise ValueError(
                "0 is not a valid refinement-query limit; use null (or omit "
                "the field) for 'no limit', or a positive integer for an "
                "explicit cap."
            )
        if value is not None and value < 0:
            raise ValueError("refinement-query limit must be positive or null")
        return value

    @property
    def max_refinement_queries_per_bidder(self) -> int:
        """Deprecated, 0-means-unlimited view of
        :attr:`per_bidder_refinement_query_limit`, for any code still
        reading the old attribute name."""
        return self.per_bidder_refinement_query_limit or 0

    @property
    def max_total_refinement_queries(self) -> int:
        """Deprecated, 0-means-unlimited view of
        :attr:`global_refinement_query_safety_limit`, for any code still
        reading the old attribute name."""
        return self.global_refinement_query_safety_limit or 0


class ExperimentArm(BaseModel):
    """One fully-specified run: one CLI invocation's worth of parameters."""

    name: str
    label: ArmLabel = "final"
    # provider/model defaults must stay a valid pairing -- "ollama" +
    # "gemini-3.1-flash-lite" previously mismatched (a Gemini model name
    # under the ollama provider), which is exactly the misleading
    # combination this schema should never default to.
    provider: str = "gemini"
    model: str = "gemini-3.1-flash-lite"
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    proxy_candidate_policy: ProxyCandidatePolicy = Field(
        default_factory=ProxyCandidatePolicy
    )
    valuation_calibration: ValuationCalibration = Field(
        default_factory=ValuationCalibration
    )
    sealed_treatment: SealedTreatment = Field(default_factory=SealedTreatment)
    clock_treatment: ClockTreatment = Field(default_factory=ClockTreatment)
    safety_limits: SafetyLimits = Field(default_factory=SafetyLimits)
    elicited_clock: bool = False
    skip_baselines: bool = False
    # Escape hatch for CLI flags this schema doesn't model yet. Keys are
    # dest-style (underscored); values `True`/`False` toggle a store_true
    # flag, anything else is passed as `--key value`.
    extra_cli_args: dict[str, Any] = Field(default_factory=dict)

    def to_cli_args(self) -> list[str]:
        """Flatten this arm into argv for ``run_live_llm_curated_batch.py``."""
        env = self.environment
        pcp = self.proxy_candidate_policy
        vc = self.valuation_calibration
        st = self.sealed_treatment
        ct = self.clock_treatment
        sl = self.safety_limits

        args: list[str] = [
            "--provider", self.provider,
            "--model", self.model,
            "--scenario", *env.scenario,
            "--seed-type", env.seed_type,
            "--num-goods", str(env.num_goods),
            "--num-bidders", str(env.num_bidders),
            "--scenario-seed", str(env.scenario_seed),
        ]
        if env.scenario_spec:
            args += ["--scenario-spec", env.scenario_spec]

        args += ["--proxy-type", pcp.proxy_type]
        if pcp.ask_initial_question:
            args.append("--ask-initial-question")
        if pcp.use_interest_map:
            args.append("--use-interest-map")
        if pcp.use_provisional_valuations:
            args.append("--use-provisional-valuations")
        args += ["--pv-max-tokens", str(pcp.pv_max_tokens)]
        if pcp.max_candidate_bundles is not None:
            args += ["--max-candidate-bundles", str(pcp.max_candidate_bundles)]
        if pcp.max_bundle_size is not None:
            args += ["--max-bundle-size", str(pcp.max_bundle_size)]
        args += ["--refinement-strategy", pcp.refinement_strategy]
        if pcp.proxy_type == "hybrid":
            args += [
                "--hybrid-alpha", str(pcp.hybrid_alpha),
                "--hybrid-delta", str(pcp.hybrid_delta),
            ]

        args += ["--epsilon", str(vc.epsilon)]
        if vc.discount_inferred:
            args.append("--discount-inferred")
        if vc.disable_anchor_values:
            args.append("--disable-anchor-values")

        args += [
            "--sealed-feedback-rule", st.sealed_feedback_rule,
            "--sealed-elicitation-rounds", str(st.sealed_elicitation_rounds),
        ]

        args += ["--top-k", *[str(k) for k in ct.top_k]]
        args += [
            "--clock-tie-threshold", str(ct.clock_tie_threshold),
            "--clock-margin-threshold", str(ct.clock_margin_threshold),
            "--price-step", str(ct.price_step),
            "--max-rounds", str(ct.max_rounds),
            "--reserve", str(ct.reserve),
        ]

        # null/omitted means "no limit" -- the CLI flag is only emitted
        # (for backwards compatibility with `examples/run_live_llm_curated_batch.py`)
        # when an explicit positive cap is actually set. This is the fix for
        # the ambiguous `--max-refinement-queries-per-bidder 0` that used to
        # appear in every dry-run command regardless of whether a cap was
        # ever intended.
        if sl.per_bidder_refinement_query_limit is not None:
            args += [
                "--max-refinement-queries-per-bidder",
                str(sl.per_bidder_refinement_query_limit),
            ]
        if sl.global_refinement_query_safety_limit is not None:
            args += [
                "--max-total-refinement-queries",
                str(sl.global_refinement_query_safety_limit),
            ]

        if self.elicited_clock:
            args.append("--elicited-clock")
        if self.skip_baselines:
            args.append("--skip-baselines")

        for key, value in self.extra_cli_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args += [flag, str(value)]

        return args


class ExperimentGrid(BaseModel):
    """A named collection of arms, all sharing one methodological purpose."""

    grid_name: str
    description: str = ""
    grid_type: Literal["value_calibration", "development", "final", "robustness"]
    arms: list[ExperimentArm]

    @model_validator(mode="after")
    def _check_unique_arm_names(self) -> "ExperimentGrid":
        names = [arm.name for arm in self.arms]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate arm names in grid {self.grid_name!r}: "
                f"{sorted(duplicates)}"
            )
        return self


def load_experiment_grid(path: str | Path) -> ExperimentGrid:
    """Parse and validate an experiment-grid JSON config file."""
    data = json.loads(Path(path).read_text())
    return ExperimentGrid.model_validate(data)


def grid_methodology_warnings(grid: ExperimentGrid) -> list[str]:
    """Non-fatal methodology warnings for a parsed grid.

    Flags settings that contradict the separation described in
    ``docs/parameter_tuning_methodology.md``: candidate caps, restrictive
    per-bidder refinement caps, and an explicit ``max_bundle_size`` are all
    legitimate *robustness/ablation* settings, but should not appear silently
    in a ``"final"``- or ``"development"``-labelled arm.
    """
    warnings: list[str] = []
    for arm in grid.arms:
        if arm.label == "robustness":
            continue
        pcp = arm.proxy_candidate_policy
        if pcp.max_candidate_bundles is not None:
            warnings.append(
                f"arm {arm.name!r} (label={arm.label!r}): max_candidate_bundles="
                f"{pcp.max_candidate_bundles} is set but the arm is not "
                "labelled 'robustness' -- candidate caps should only appear "
                "in explicitly-labelled robustness/ablation arms."
            )
        if pcp.max_bundle_size is not None:
            warnings.append(
                f"arm {arm.name!r} (label={arm.label!r}): max_bundle_size="
                f"{pcp.max_bundle_size} is set but the arm is not labelled "
                "'robustness' -- treat max_bundle_size as an explicit "
                "ablation, not a silently-tuned default."
            )
        if arm.safety_limits.per_bidder_refinement_query_limit is not None:
            warnings.append(
                f"arm {arm.name!r} (label={arm.label!r}): "
                "per_bidder_refinement_query_limit="
                f"{arm.safety_limits.per_bidder_refinement_query_limit} is set "
                "but the arm is not labelled 'robustness' -- refinement count "
                "should be an outcome of the elicitation events and "
                "mechanism, not a low binding cap."
            )
        if (
            arm.sealed_treatment.sealed_elicitation_rounds > 0
            and arm.sealed_treatment.sealed_feedback_rule == "none"
        ):
            warnings.append(
                f"arm {arm.name!r}: sealed_elicitation_rounds > 0 with "
                "sealed_feedback_rule='none' is a no-op -- the proxy never "
                "receives refinement feedback."
            )
    return warnings
