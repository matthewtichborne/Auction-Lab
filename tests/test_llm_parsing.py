from __future__ import annotations

import pytest

from auctionlab.llm.parsing import (
    parse_value_response,
    validate_queried_bundle,
)
from auctionlab.llm.schemas import LlmValueResponse


def test_json_value_response_parses():
    assert parse_value_response('{"value": 612}') == LlmValueResponse(
        bundle_value=612
    )


def test_json_value_response_with_queried_bundle_parses():
    parsed = parse_value_response(
        '{"queried_bundle": ["IPAD"], "bundle_value": 500}'
    )

    assert parsed.queried_bundle == ["IPAD"]
    assert parsed.bundle_value == 500.0


def test_json_value_response_with_anchor_decomposition_parses():
    parsed = parse_value_response(
        """
        {
          "queried_bundle": ["CAMERA", "LENS"],
          "base_value_from_anchors": 500,
          "synergy_adjustment": 350,
          "bundle_value": 850,
          "confidence": 0.8,
          "reasoning_summary": "The pair forms a complete setup."
        }
        """
    )

    assert parsed.base_value_from_anchors == 500.0
    assert parsed.synergy_adjustment == 350.0
    assert parsed.bundle_value == 850.0


def test_old_json_without_anchor_decomposition_still_parses():
    parsed = parse_value_response('{"bundle_value": 500}')

    assert parsed.base_value_from_anchors is None
    assert parsed.synergy_adjustment is None
    assert parsed.bundle_value == 500.0


def test_validate_queried_bundle_accepts_matching_bundle_in_any_order():
    parsed = LlmValueResponse(
        queried_bundle=["PENCIL", "IPAD"],
        bundle_value=650,
    )

    validate_queried_bundle(
        parsed,
        frozenset({"IPAD", "PENCIL"}),
    )


def test_validate_queried_bundle_rejects_wrong_bundle():
    parsed = LlmValueResponse(
        queried_bundle=["IPAD", "PENCIL"],
        bundle_value=650,
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_queried_bundle(parsed, frozenset({"IPAD"}))


def test_validate_queried_bundle_rejects_duplicates():
    parsed = LlmValueResponse(
        queried_bundle=["IPAD", "IPAD"],
        bundle_value=500,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_queried_bundle(parsed, frozenset({"IPAD"}))


def test_validate_queried_bundle_rejects_non_string_items():
    parsed = LlmValueResponse.model_construct(
        queried_bundle=["IPAD", 1],
        bundle_value=500.0,
        confidence=None,
        reasoning_summary=None,
    )

    with pytest.raises(ValueError, match="string"):
        validate_queried_bundle(parsed, frozenset({"IPAD"}))


def test_validate_queried_bundle_is_no_op_when_missing():
    validate_queried_bundle(
        LlmValueResponse(bundle_value=500),
        frozenset({"IPAD"}),
    )


def test_markdown_fenced_json_value_response_parses():
    raw = """Here is the result:
```json
{"value": 612.5}
```
"""

    assert parse_value_response(raw) == LlmValueResponse(bundle_value=612.5)


def test_bundle_value_text_parses():
    assert parse_value_response("Bundle value: $612") == LlmValueResponse(
        bundle_value=612
    )


def test_bundle_value_text_with_is_parses():
    assert parse_value_response(
        "The bundle value is $240.00"
    ) == LlmValueResponse(bundle_value=240.0)


@pytest.mark.parametrize(
    "raw",
    [
        '{"value": -1}',
        "Bundle value: $-1",
    ],
)
def test_negative_values_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_value_response(raw)
