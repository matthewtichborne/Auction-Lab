from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field


class LlmValueResponse(BaseModel):
    """Direct value-query response.

    ``queried_bundle`` is no longer requested in the prompt -- the bundle is
    fixed by the caller, not chosen by the model -- but is accepted for
    backward compatibility with older responses/tests. When present it is
    strictly validated against the caller's expected bundle (see
    :func:`auctionlab.llm.parsing.validate_queried_bundle`); when absent
    (the preferred, minimal response) validation is a no-op.
    """

    queried_bundle: list[str] | None = None
    base_value_from_anchors: float | None = Field(default=None, ge=0.0)
    synergy_adjustment: float | None = None
    bundle_value: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("bundle_value", "value"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning_summary: str | None = None

    @property
    def value(self) -> float:
        return self.bundle_value


class LlmAtomicBidResponse(BaseModel):
    bundle: list[str]
    value: float = Field(ge=0.0)


class LlmDemandResponse(BaseModel):
    primary_bundle: list[str] | None
    supplementary_atoms: list[LlmAtomicBidResponse]


class LlmNaturalLanguageAnswer(BaseModel):
    answer: str


class LlmItemDisclosure(BaseModel):
    """One item-level semantic claim extracted from a person answer."""

    item_id: str
    evidence: str = Field(min_length=1)


class LlmComplementGroupEvidence(BaseModel):
    """Evidence that a complete set has value beyond its separate members."""

    items: list[str] = Field(min_length=2)
    evidence: str = Field(min_length=1)
    explicit_extra_joint_value: bool


class LlmComplementEntailmentJudgment(BaseModel):
    """Independent text-only check of one proposed complement group."""

    items: list[str] = Field(min_length=2)
    entailed: bool
    evidence: str = ""
    reason: str = ""


class LlmComplementEntailmentResponse(BaseModel):
    judgments: list[LlmComplementEntailmentJudgment] = Field(
        default_factory=list
    )


class LlmSubstituteModeEntailmentJudgment(BaseModel):
    """Focused check that an answer explicitly states an acquisition mode."""

    items: list[str] = Field(min_length=2)
    acquisition_mode: Literal["choose_one", "can_use_multiple"]
    entailed: bool
    evidence: str = ""
    reason: str = ""


class LlmSubstituteModeEntailmentResponse(BaseModel):
    judgments: list[LlmSubstituteModeEntailmentJudgment] = Field(
        default_factory=list
    )


class LlmPersonAnswerSemanticExtraction(BaseModel):
    """Blind reconstruction of the economic content of a person answer."""

    positive_items: list[LlmItemDisclosure] = Field(default_factory=list)
    excluded_items: list[LlmItemDisclosure] = Field(default_factory=list)
    substitute_groups: list["LlmExtractedSubstituteGroup"] = Field(
        default_factory=list
    )
    complementary_groups: list[LlmComplementGroupEvidence] = Field(
        default_factory=list
    )
    budget_hint: float | None = Field(default=None, ge=0.0)
    other_numeric_valuation_details: list[str] = Field(default_factory=list)


class LlmPersonAnswerVerification(BaseModel):
    """Truth comparison derived from a blind semantic extraction."""

    passed: bool
    missing_positive_items: list[str] = Field(default_factory=list)
    incorrectly_excluded_positive_items: list[str] = Field(
        default_factory=list
    )
    invented_positive_items: list[str] = Field(default_factory=list)
    substitute_group_issues: list[str] = Field(default_factory=list)
    complement_group_issues: list[str] = Field(default_factory=list)
    budget_preserved: bool = True
    invented_numeric_detail: bool = False
    issues: list[str] = Field(default_factory=list)
    repair_instructions: str = ""
    semantic_extraction: dict | None = None


class LlmDemandQueryResponse(BaseModel):
    satisfied: bool
    preferred_bundle: list[str] | None = None


class LlmProxyQuestionResponse(BaseModel):
    question: str


class LlmSummaryUpdateResponse(BaseModel):
    summary: str


class LlmProvisionalValuationEntry(BaseModel):
    bundle: list[str]
    value: float = Field(ge=0.0)


class LlmProvisionalValuations(BaseModel):
    valuations: list[LlmProvisionalValuationEntry]
    reasoning: str


class LlmCompactProvisionalValuations(BaseModel):
    """Ordered PV response; positions correspond to the requested bundles."""

    values: list[Annotated[float, Field(ge=0.0)]]
    reasoning: str = ""


class LlmSubstituteGroup(BaseModel):
    """Proxy-inferred, evidence-backed relationship between alternatives."""

    items: list[str] = Field(min_length=2)
    acquisition_mode: Literal[
        "choose_one", "can_use_multiple", "unclear"
    ]
    evidence: str = Field(
        min_length=1,
        description="Short evidence from the person's answer.",
    )
    mode_explicitly_stated: bool | None = Field(
        default=None,
        description=(
            "Whether the answer explicitly establishes this acquisition "
            "mode. None is retained only for legacy artefacts."
        ),
    )


class LlmExtractedSubstituteGroup(BaseModel):
    """Truth-blind person-answer extraction with mode explicitness."""

    items: list[str] = Field(min_length=2)
    acquisition_mode: Literal[
        "choose_one", "can_use_multiple", "unclear"
    ]
    evidence: str = Field(min_length=1)
    mode_explicitly_stated: bool = False


class LlmInterestMap(BaseModel):
    """Structured interest map extracted from a person's NL answer.

    Derived by the proxy from the person's answer to the initial preference
    question. Used to filter and prioritise candidate bundles before any
    value queries are issued.
    """

    interested_items: list[str] = Field(
        description="Item IDs the person has positive interest in acquiring.",
    )
    excluded_items: list[str] = Field(
        default_factory=list,
        description="Item IDs the person has zero or negligible interest in.",
    )
    complementary_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Item sets the person values highly as complete bundles. "
            "Each group is included as a bundle at the highest priority."
        ),
    )
    complementary_group_evidence: list[LlmComplementGroupEvidence] = Field(
        default_factory=list,
        description=(
            "Evidence records for proposed complementary groups. Newly "
            "inferred groups are accepted only when a matching record says "
            "the answer explicitly claims extra complete-set value."
        ),
    )
    substitute_groups: list[LlmSubstituteGroup] = Field(
        default_factory=list,
        description=(
            "Person-specific related alternatives with an inferred "
            "acquisition mode and textual evidence. Only explicit "
            "choose_one groups are safe for hard candidate filtering."
        ),
    )
    budget_hint: float | None = Field(
        default=None,
        description="Approximate total willingness to pay, if stated.",
    )
    reasoning: str = Field(
        description="Brief explanation of the extraction rationale.",
    )


LateReflectionTargetType = Literal[
    "large_bundle_check",
    "complementarity_check",
    "substitute_check",
    "missing_bundle",
    "final_allocation_acceptability",
    "other",
]

LateReflectionFollowupType = Literal["none", "value_query", "demand_query"]

LateReflectionMode = Literal["bundle_comparison", "marginal_item_test"]


class LlmLateReflectionResponse(BaseModel):
    """Structured output of the late-stage reflective elicitation LLM call.

    Distinct from :class:`LlmProxyQuestionResponse` (the opening ωnvd
    question): this call happens near auction termination, with full
    accumulated context, and asks for exactly one targeted clarification
    plus a machine-actionable follow-up suggestion.

    Strict prompt, forgiving parser: the *prompt* still asks for
    ``reflection_mode`` (see :func:`auctionlab.llm.prompts.build_late_reflection_prompt`)
    and a ``bundle_comparison``/``marginal_item_test`` pairwise structure,
    but this schema deliberately does **not** hard-fail the whole row when
    ``reflection_mode``/``target_type``/``suggested_followup`` are missing,
    misspelled, or use an old-style (pre-pairwise) response shape -- a live
    10x10 run showed a strict ``Literal``-typed ``reflection_mode`` turned
    truncated/old-style output into a 100% parse-failure rate, silently
    reverting every late-reflection arm to the non-reflection baseline.
    Recovery happens in :func:`auctionlab.llm.parsing.parse_late_reflection_response`
    (bundle-based ``reflection_mode`` inference,
    ``primary_bundle``/``comparison_bundle`` derived from
    ``followup_bundles``/``target_bundles`` and vice versa); this class only
    declares the permissive shape recovery needs to work with, plus
    ``reflection_mode_inferred`` so downstream logging can tell an
    LLM-declared mode from a parser-inferred one.

    ``bundle_comparison`` compares ``primary_bundle`` against
    ``comparison_bundle`` (current allocation/demand vs. a rejected
    alternative, near-tie bundle, or substitute); ``marginal_item_test``
    tests whether ``marginal_item`` matters, framed as ``primary_bundle``
    (without the item) vs. ``comparison_bundle`` (with it), or a large
    bundle vs. its core subset.
    """

    question: str = Field(min_length=1)
    reason: str = ""
    # Not a strict Literal: an unrecognised/missing value is recovered by
    # the parser rather than raising -- see module docstring above.
    reflection_mode: str | None = None
    reflection_mode_inferred: bool = False
    target_type: str = "other"
    primary_bundle: list[str] | None = None
    comparison_bundle: list[str] | None = None
    marginal_item: str | None = None
    target_bundles: list[list[str]] = Field(default_factory=list)
    suggested_followup: str = "none"
    followup_bundles: list[list[str]] = Field(default_factory=list)
