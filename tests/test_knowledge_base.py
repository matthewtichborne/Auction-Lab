"""Tests for KnowledgeBase protocol, TranscriptKnowledgeBase, and SummaryKnowledgeBase."""

from __future__ import annotations

from auctionlab.llm.clients import MockLlmClient
from auctionlab.llm.knowledge import KnowledgeBase, SummaryKnowledgeBase, TranscriptKnowledgeBase


def test_transcript_knowledge_base_is_empty_by_default():
    kb = TranscriptKnowledgeBase()
    assert not kb
    assert len(kb) == 0
    assert kb.context_for_prompt() == ""


def test_add_qa_makes_knowledge_base_non_empty():
    kb = TranscriptKnowledgeBase()
    kb.add_qa("What matters most?", "Portability.")
    assert kb
    assert len(kb) == 1


def test_context_for_prompt_includes_question_and_answer():
    kb = TranscriptKnowledgeBase()
    kb.add_qa("What matters?", "Speed.")
    context = kb.context_for_prompt()
    assert "PRIOR_PREFERENCE_QA:" in context
    assert "What matters?" in context
    assert "Speed." in context


def test_entries_returns_snapshot_not_internal_reference():
    kb = TranscriptKnowledgeBase()
    kb.add_qa("Q1", "A1")
    entries = kb.entries
    entries.append(("Q2", "A2"))  # mutating the snapshot should not affect kb
    assert len(kb) == 1


def test_multiple_qa_pairs_all_appear_in_context():
    kb = TranscriptKnowledgeBase()
    kb.add_qa("First question", "First answer")
    kb.add_qa("Second question", "Second answer")
    context = kb.context_for_prompt()
    assert "First question" in context
    assert "First answer" in context
    assert "Second question" in context
    assert "Second answer" in context
    assert len(kb) == 2


def test_transcript_knowledge_base_satisfies_protocol():
    kb = TranscriptKnowledgeBase()
    assert isinstance(kb, KnowledgeBase)


def test_knowledge_base_shares_backing_list_with_nl_transcript():
    """TranscriptKnowledgeBase wrapping a list stays in sync with it."""
    nl_transcript: list[tuple[str, str]] = []
    kb = TranscriptKnowledgeBase(nl_transcript)

    kb.add_qa("Q1", "A1")

    # The original list is mutated by the knowledge base.
    assert nl_transcript == [("Q1", "A1")]
    # And changes to the list are reflected in the knowledge base.
    nl_transcript.append(("Q2", "A2"))
    assert len(kb) == 2


def test_proxy_knowledge_base_initialised_on_construction():
    """LlmInferredXorProxy should expose a knowledge_base after construction."""
    from auctionlab.llm.clients import MockLlmClient
    from auctionlab.llm.person_simulator import LlmPersonSimulator
    from auctionlab.llm.proxies import LlmInferredXorProxy

    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test",
        person_seed="Values A.",
        item_descriptions={"A": "Item A"},
        client=MockLlmClient([]),
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person)

    assert isinstance(proxy.knowledge_base, KnowledgeBase)
    assert not proxy.knowledge_base


def test_ask_initial_question_updates_knowledge_base():
    """ask_initial_question should add a Q/A pair via the knowledge base."""
    from auctionlab.llm.clients import MockLlmClient
    from auctionlab.llm.person_simulator import LlmPersonSimulator
    from auctionlab.llm.proxies import LlmInferredXorProxy

    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test",
        person_seed="Values A.",
        item_descriptions={"A": "Item A"},
        client=MockLlmClient(
            [
                '{"question": "What matters most to you?"}',
                '{"answer": "Durability above all."}',
            ]
        ),
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person)
    proxy.ask_initial_question()

    # Both the knowledge_base and nl_transcript (its backing list) are updated.
    assert len(proxy.knowledge_base) == 1
    assert proxy.nl_transcript == [
        ("What matters most to you?", "Durability above all.")
    ]
    assert "PRIOR_PREFERENCE_QA:" in proxy.knowledge_base.context_for_prompt()


# ---------------------------------------------------------------------------
# SummaryKnowledgeBase tests
# ---------------------------------------------------------------------------


def _make_summary_kb(responses: list[str]) -> SummaryKnowledgeBase:
    return SummaryKnowledgeBase(client=MockLlmClient(responses))


def test_summary_knowledge_base_is_empty_by_default():
    kb = _make_summary_kb([])
    assert not kb
    assert kb.summary == ""
    assert kb.recent_qa == []
    assert kb.context_for_prompt() == ""


def test_summary_knowledge_base_satisfies_protocol():
    kb = _make_summary_kb([])
    assert isinstance(kb, KnowledgeBase)


def test_summary_knowledge_base_add_qa_updates_summary():
    kb = _make_summary_kb(['{"summary": "Person values speed."}'])
    kb.add_qa("What do you care about?", "Mostly speed.")
    assert kb.summary == "Person values speed."
    assert kb


def test_summary_knowledge_base_add_qa_keeps_recent_pair():
    kb = _make_summary_kb(['{"summary": "Prefers A."}'])
    kb.add_qa("Favourite item?", "Item A.")
    assert kb.recent_qa == [("Favourite item?", "Item A.")]


def test_summary_knowledge_base_context_for_prompt_includes_summary():
    kb = _make_summary_kb(['{"summary": "Values portability."}'])
    kb.add_qa("Q", "A")
    context = kb.context_for_prompt()
    assert "PREFERENCE_SUMMARY:" in context
    assert "Values portability." in context


def test_summary_knowledge_base_context_includes_recent_qa():
    kb = _make_summary_kb(['{"summary": "S"}'])
    kb.add_qa("Q?", "A!")
    context = kb.context_for_prompt()
    assert "RECENT_QA:" in context
    assert "Q?" in context
    assert "A!" in context


def test_summary_knowledge_base_recent_qa_window_is_respected():
    responses = [
        '{"summary": "S1"}',
        '{"summary": "S2"}',
        '{"summary": "S3"}',
        '{"summary": "S4"}',
    ]
    kb = SummaryKnowledgeBase(client=MockLlmClient(responses), max_recent=2)
    kb.add_qa("Q1", "A1")
    kb.add_qa("Q2", "A2")
    kb.add_qa("Q3", "A3")
    kb.add_qa("Q4", "A4")
    # Only the 2 most recent pairs are kept.
    assert kb.recent_qa == [("Q3", "A3"), ("Q4", "A4")]


def test_summary_knowledge_base_rolling_summary_updated_each_call():
    responses = [
        '{"summary": "First summary."}',
        '{"summary": "Second summary."}',
    ]
    kb = _make_summary_kb(responses)
    kb.add_qa("Q1", "A1")
    assert kb.summary == "First summary."
    kb.add_qa("Q2", "A2")
    assert kb.summary == "Second summary."


def test_summary_knowledge_base_recent_qa_is_snapshot():
    kb = _make_summary_kb(['{"summary": "S"}'])
    kb.add_qa("Q", "A")
    snapshot = kb.recent_qa
    snapshot.append(("X", "Y"))
    assert kb.recent_qa == [("Q", "A")]  # internal list unchanged


def test_summary_knowledge_base_raises_on_bad_llm_response():
    kb = _make_summary_kb(["not valid json"])
    import pytest
    with pytest.raises(ValueError):
        kb.add_qa("Q", "A")


def test_summary_knowledge_base_bool_true_with_only_summary():
    kb = _make_summary_kb(['{"summary": "Something."}'])
    kb.add_qa("Q", "A")
    # Manually clear recent to test summary-only bool
    kb._recent_qa.clear()
    assert kb


def test_summary_knowledge_base_bool_true_with_only_recent_qa():
    # Inject a summary KB with pre-set recent_qa but empty summary
    kb = _make_summary_kb([])
    kb._recent_qa.append(("Q", "A"))
    assert kb


# ---------------------------------------------------------------------------


def test_knowledge_base_context_flows_into_value_query_prompt():
    """Knowledge base context should appear in LLM prompts for value queries."""
    from auctionlab.llm.clients import MockLlmClient
    from auctionlab.llm.person_simulator import LlmPersonSimulator
    from auctionlab.llm.proxies import LlmInferredXorProxy

    person = LlmPersonSimulator(
        bidder_id="b1",
        scenario_description="Test",
        person_seed="Values A.",
        item_descriptions={"A": "Item A"},
        client=MockLlmClient(
            [
                '{"question": "What matters?"}',
                '{"answer": "Speed is everything."}',
                '{"bundle_value": 100}',
            ]
        ),
    )
    proxy = LlmInferredXorProxy(bidder_id="b1", person=person)
    proxy.ask_initial_question()
    proxy.infer_xor_bid([frozenset({"A"})])

    value_query_prompt = person.client.calls[-1]
    assert "PRIOR_PREFERENCE_QA:" in value_query_prompt
    assert "Speed is everything." in value_query_prompt
