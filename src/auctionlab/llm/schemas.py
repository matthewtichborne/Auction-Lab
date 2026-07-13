from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class LlmValueResponse(BaseModel):
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
    substitute_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Item groups where the person wants at most one item. Bundles "
            "containing two or more items from the same group are excluded."
        ),
    )
    budget_hint: float | None = Field(
        default=None,
        description="Approximate total willingness to pay, if stated.",
    )
    reasoning: str = Field(
        description="Brief explanation of the extraction rationale.",
    )
