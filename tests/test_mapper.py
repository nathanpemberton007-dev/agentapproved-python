"""Tests for EU AI Act Article 12 compliance mapper."""

import json
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from agentapproved import AgentApprovedHandler, assess_compliance


# ── Helper: Build handler with specific event patterns ──────────


def _handler_with_full_agent_session() -> AgentApprovedHandler:
    """Simulate a complete agent session: LLM + tool + RAG + oversight."""
    h = AgentApprovedHandler(agent_id="full-agent", actor_id="user-1")
    run_id = uuid4()

    # LLM call
    h.on_llm_start(
        serialized={"name": "LLM", "kwargs": {"model_name": "gpt-4o"}},
        prompts=["What is the return policy?"],
        run_id=run_id,
    )
    h.on_llm_end(
        response=LLMResult(
            generations=[[Generation(text="Let me check the policy.")]],
            llm_output=None,
        ),
        run_id=run_id,
    )

    # Tool call
    tool_run = uuid4()
    h.on_tool_start(
        serialized={"name": "PolicySearch"},
        input_str="return policy",
        run_id=tool_run,
    )
    h.on_tool_end(output="30 day returns accepted.", run_id=tool_run)

    # Retrieval
    ret_run = uuid4()
    h.on_retriever_start(
        serialized={"name": "VectorRetriever"},
        query="return policy details",
        run_id=ret_run,
    )
    h.on_retriever_end(
        documents=[
            Document(
                page_content="Returns accepted within 30 days.",
                metadata={"source": "policy.pdf"},
            ),
        ],
        run_id=ret_run,
    )

    # Agent decision + finish
    h.on_agent_action(
        action=AgentAction(tool="respond", tool_input="summary", log="Summarising"),
        run_id=uuid4(),
    )
    h.on_agent_finish(
        finish=AgentFinish(return_values={"output": "You can return within 30 days."}, log="Done"),
        run_id=uuid4(),
    )

    # Human oversight
    h.record_oversight(
        reviewer_id="auditor-jane",
        decision="approved",
        reason="Accurate response",
    )

    # Session end
    h.end_session()
    return h


def _handler_llm_only() -> AgentApprovedHandler:
    """Session with LLM calls only — no RAG, no tools, no oversight."""
    h = AgentApprovedHandler(agent_id="llm-only")
    run_id = uuid4()
    h.on_llm_start(
        serialized={"name": "LLM", "kwargs": {}},
        prompts=["Hello"],
        run_id=run_id,
    )
    h.on_llm_end(
        response=LLMResult(generations=[[Generation(text="Hi")]], llm_output=None),
        run_id=run_id,
    )
    h.end_session()
    return h


def _handler_no_oversight() -> AgentApprovedHandler:
    """Full session but missing human oversight."""
    h = AgentApprovedHandler(agent_id="no-oversight")
    run_id = uuid4()

    h.on_llm_start(
        serialized={"name": "LLM", "kwargs": {}},
        prompts=["Query"],
        run_id=run_id,
    )
    h.on_llm_end(
        response=LLMResult(generations=[[Generation(text="Response")]], llm_output=None),
        run_id=run_id,
    )

    ret_run = uuid4()
    h.on_retriever_start(serialized={"name": "Ret"}, query="search", run_id=ret_run)
    h.on_retriever_end(
        documents=[Document(page_content="Result", metadata={"source": "db"})],
        run_id=ret_run,
    )

    h.on_agent_action(
        action=AgentAction(tool="act", tool_input="x", log="log"),
        run_id=uuid4(),
    )
    h.on_agent_finish(
        finish=AgentFinish(return_values={"output": "done"}, log="done"),
        run_id=uuid4(),
    )

    h.end_session()
    return h


# ── Overall Score ───────────────────────────────────────────────


class TestOverallScore:
    def test_full_session_scores_100(self):
        h = _handler_with_full_agent_session()
        report = assess_compliance(h.get_events())
        assert report.overall_score == 100
        assert report.missing == 0
        assert report.satisfied == 6

    def test_empty_events_scores_0(self):
        report = assess_compliance([])
        assert report.overall_score == 0
        assert report.missing > 0

    def test_llm_only_session_score(self):
        h = _handler_llm_only()
        report = assess_compliance(h.get_events())
        # Art 12(1) satisfied, 12(2)(a) satisfied, 12(2)(b) N/A,
        # 12(2)(c) N/A, 12(2)(d) missing, 12(3) partial
        assert 0 < report.overall_score < 100
        assert report.not_applicable >= 2  # No RAG = 12(2)(b) + 12(2)(c) N/A

    def test_no_oversight_reduces_score(self):
        h = _handler_no_oversight()
        report = assess_compliance(h.get_events())
        assert report.overall_score < 100
        # Find the oversight requirement
        oversight = next(r for r in report.requirements if r.id == "ART_12_2_D")
        assert oversight.status == "missing"

    def test_score_is_integer(self):
        h = _handler_llm_only()
        report = assess_compliance(h.get_events())
        assert isinstance(report.overall_score, int)

    def test_score_bounded_0_to_100(self):
        for handler_fn in [_handler_with_full_agent_session, _handler_llm_only, _handler_no_oversight]:
            report = assess_compliance(handler_fn().get_events())
            assert 0 <= report.overall_score <= 100


# ── Individual Article Checks ───────────────────────────────────


class TestArticle12_1:
    """Automatic logging capability."""

    def test_satisfied_when_events_exist(self):
        h = AgentApprovedHandler()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_1")
        assert art.status == "satisfied"
        assert art.evidence_count > 0

    def test_missing_when_no_events(self):
        report = assess_compliance([])
        art = next(r for r in report.requirements if r.id == "ART_12_1")
        assert art.status == "missing"


class TestArticle12_2a:
    """Period of each use."""

    def test_satisfied_with_start_and_end(self):
        h = AgentApprovedHandler()
        h.end_session()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_A")
        assert art.status == "satisfied"

    def test_partial_with_start_only(self):
        h = AgentApprovedHandler()
        # No end_session
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_A")
        assert art.status == "partial"


class TestArticle12_2b:
    """Reference database."""

    def test_satisfied_with_retrieval_sources(self):
        h = AgentApprovedHandler()
        run_id = uuid4()
        h.on_retriever_start(serialized={"name": "Ret"}, query="q", run_id=run_id)
        h.on_retriever_end(
            documents=[Document(page_content="text", metadata={"source": "db.pdf"})],
            run_id=run_id,
        )
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_B")
        assert art.status == "satisfied"

    def test_not_applicable_when_no_retrieval(self):
        h = AgentApprovedHandler()
        h.end_session()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_B")
        assert art.status == "not_applicable"


class TestArticle12_2c:
    """Input data leading to match."""

    def test_satisfied_with_query_and_results(self):
        h = AgentApprovedHandler()
        run_id = uuid4()
        h.on_retriever_start(serialized={"name": "R"}, query="policy", run_id=run_id)
        h.on_retriever_end(
            documents=[Document(page_content="result", metadata={"source": "s"})],
            run_id=run_id,
        )
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_C")
        assert art.status == "satisfied"

    def test_not_applicable_when_no_retrieval(self):
        h = AgentApprovedHandler()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_C")
        assert art.status == "not_applicable"


class TestArticle12_2d:
    """Human oversight verification."""

    def test_satisfied_with_oversight(self):
        h = AgentApprovedHandler()
        h.record_oversight(reviewer_id="jane", decision="approved")
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_D")
        assert art.status == "satisfied"
        assert art.evidence_count > 0

    def test_missing_without_oversight(self):
        h = AgentApprovedHandler()
        h.end_session()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_2_D")
        assert art.status == "missing"
        assert art.remediation is not None
        assert "record_oversight" in art.remediation


class TestArticle12_3:
    """Post-market monitoring traceability."""

    def test_satisfied_with_diverse_events(self):
        h = _handler_with_full_agent_session()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_3")
        assert art.status == "satisfied"

    def test_partial_with_llm_only(self):
        h = _handler_llm_only()
        report = assess_compliance(h.get_events())
        art = next(r for r in report.requirements if r.id == "ART_12_3")
        assert art.status == "partial"

    def test_missing_with_no_lifecycle_events(self):
        report = assess_compliance([])
        art = next(r for r in report.requirements if r.id == "ART_12_3")
        assert art.status == "missing"


# ── Report Structure ────────────────────────────────────────────


class TestReportStructure:
    def test_report_has_all_six_requirements(self):
        h = AgentApprovedHandler()
        report = assess_compliance(h.get_events())
        assert len(report.requirements) == 6
        ids = {r.id for r in report.requirements}
        assert ids == {
            "ART_12_1", "ART_12_2_A", "ART_12_2_B",
            "ART_12_2_C", "ART_12_2_D", "ART_12_3",
        }

    def test_report_to_dict(self):
        h = AgentApprovedHandler()
        report = assess_compliance(h.get_events())
        d = report.to_dict()
        assert d["regulation"] == "EU AI Act"
        assert d["article"] == "Article 12"
        assert isinstance(d["overall_score"], int)
        assert isinstance(d["requirements"], list)
        assert len(d["requirements"]) == 6

    def test_gaps_returns_only_problems(self):
        h = _handler_no_oversight()
        report = assess_compliance(h.get_events())
        gaps = report.gaps
        assert all(g.status in ("missing", "partial") for g in gaps)
        assert any(g.id == "ART_12_2_D" for g in gaps)

    def test_remediation_present_for_gaps(self):
        h = _handler_llm_only()
        report = assess_compliance(h.get_events())
        for gap in report.gaps:
            assert gap.remediation is not None
            assert len(gap.remediation) > 10  # Not empty placeholder

    def test_counts_add_up(self):
        h = _handler_with_full_agent_session()
        report = assess_compliance(h.get_events())
        total = report.satisfied + report.missing + report.partial + report.not_applicable
        assert total == len(report.requirements)

    def test_evidence_samples_are_event_ids(self):
        h = _handler_with_full_agent_session()
        report = assess_compliance(h.get_events())
        event_ids = {e.event_id for e in h.get_events()}
        for req in report.requirements:
            for sample_id in req.evidence_sample:
                assert sample_id in event_ids
