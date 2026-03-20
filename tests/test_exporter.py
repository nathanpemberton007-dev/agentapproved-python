"""Tests for evidence packet generator — the auditor-facing export."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from agentapproved import AgentApprovedHandler, generate_evidence_packet
from agentapproved.hasher import get_public_key_hex


# ── Helper ──────────────────────────────────────────────────────


def _full_session_handler(data_dir: Path | None = None) -> AgentApprovedHandler:
    """Create a handler with a realistic agent session."""
    h = AgentApprovedHandler(agent_id="demo-agent", actor_id="user-1", data_dir=data_dir)
    run_id = uuid4()

    h.on_llm_start(
        serialized={"name": "LLM", "kwargs": {"model_name": "gpt-4o"}},
        prompts=["What is the return policy?"],
        run_id=run_id,
    )
    h.on_llm_end(
        response=LLMResult(
            generations=[[Generation(text="Let me look that up.")]],
            llm_output=None,
        ),
        run_id=run_id,
    )

    tool_run = uuid4()
    h.on_tool_start(serialized={"name": "PolicySearch"}, input_str="return policy", run_id=tool_run)
    h.on_tool_end(output="30 day returns.", run_id=tool_run)

    ret_run = uuid4()
    h.on_retriever_start(serialized={"name": "VectorDB"}, query="return policy", run_id=ret_run)
    h.on_retriever_end(
        documents=[Document(page_content="Returns within 30 days.", metadata={"source": "policy.pdf"})],
        run_id=ret_run,
    )

    h.on_agent_action(
        action=AgentAction(tool="respond", tool_input="summary", log="Summarising"),
        run_id=uuid4(),
    )
    h.on_agent_finish(
        finish=AgentFinish(return_values={"output": "You can return within 30 days."}, log="Done"),
        run_id=uuid4(),
    )

    h.record_oversight(reviewer_id="auditor-jane", decision="approved", reason="Accurate")
    h.end_session()
    return h


# ── Packet Generation ───────────────────────────────────────────


class TestPacketGeneration:
    def test_generates_three_files(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(
            h.get_events(), out,
            organisation="Test Corp",
            public_key_hex=get_public_key_hex(h._public_key),
        )
        assert (out / "evidence.json").exists()
        assert (out / "integrity.json").exists()
        assert (out / "report.html").exists()

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "deep" / "nested" / "packet"
        generate_evidence_packet(h.get_events(), out)
        assert out.is_dir()
        assert (out / "evidence.json").exists()

    def test_returns_output_path(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        result = generate_evidence_packet(h.get_events(), out)
        assert result == out


# ── Evidence JSON ───────────────────────────────────────────────


class TestEvidenceJSON:
    def test_structure(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(
            h.get_events(), out,
            organisation="Acme Ltd",
            public_key_hex=get_public_key_hex(h._public_key),
        )

        with open(out / "evidence.json", "r") as f:
            doc = json.load(f)

        assert doc["format"] == "agentapproved-evidence-v1"
        assert doc["organisation"] == "Acme Ltd"
        assert "generated_at" in doc
        assert "period" in doc
        assert doc["period"]["start"] != ""
        assert doc["period"]["end"] != ""
        assert doc["public_key"] != ""
        assert doc["event_count"] == len(h.get_events())
        assert len(doc["events"]) == doc["event_count"]

    def test_compliance_report_embedded(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out)

        with open(out / "evidence.json", "r") as f:
            doc = json.load(f)

        compliance = doc["compliance"]
        assert compliance["regulation"] == "EU AI Act"
        assert compliance["overall_score"] == 100
        assert len(compliance["requirements"]) == 6

    def test_events_match_handler(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out)

        with open(out / "evidence.json", "r") as f:
            doc = json.load(f)

        original_ids = [e.event_id for e in h.get_events()]
        exported_ids = [e["event_id"] for e in doc["events"]]
        assert original_ids == exported_ids


# ── Integrity JSON ──────────────────────────────────────────────


class TestIntegrityJSON:
    def test_structure(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(
            h.get_events(), out,
            public_key_hex=get_public_key_hex(h._public_key),
        )

        with open(out / "integrity.json", "r") as f:
            doc = json.load(f)

        assert doc["format"] == "agentapproved-integrity-v1"
        assert doc["chain_valid"] is True
        assert doc["event_count"] == len(h.get_events())
        assert doc["chain_start"] == "GENESIS"
        assert len(doc["chain_end"]) == 64
        assert doc["signatures_present"] is True
        assert len(doc["events_summary"]) == doc["event_count"]

    def test_events_summary_has_truncated_signatures(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(
            h.get_events(), out,
            public_key_hex=get_public_key_hex(h._public_key),
        )

        with open(out / "integrity.json", "r") as f:
            doc = json.load(f)

        for summary in doc["events_summary"]:
            assert summary["signature"].endswith("...")
            assert "event_hash" in summary
            assert "action_type" in summary


# ── HTML Report ─────────────────────────────────────────────────


class TestHTMLReport:
    def test_is_valid_html(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out, organisation="Test Corp")

        html = (out / "report.html").read_text(encoding="utf-8")
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "AgentApproved" in html

    def test_contains_organisation_name(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out, organisation="PowerBee Ltd")

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "PowerBee Ltd" in html

    def test_contains_compliance_score(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out)

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "100%" in html

    def test_contains_all_six_articles(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out)

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "Article 12(1)" in html
        assert "Article 12(2)(a)" in html
        assert "Article 12(2)(b)" in html
        assert "Article 12(2)(c)" in html
        assert "Article 12(2)(d)" in html
        assert "Article 12(3)" in html

    def test_contains_chain_verification(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(
            h.get_events(), out,
            public_key_hex=get_public_key_hex(h._public_key),
        )

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "VERIFIED" in html
        assert "GENESIS" in html

    def test_shows_remediation_for_gaps(self, tmp_path: Path):
        """Handler with no oversight — report should show remediation."""
        h = AgentApprovedHandler(agent_id="no-oversight")
        h.end_session()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out)

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "Remediation" in html
        assert "record_oversight" in html

    def test_html_escapes_special_characters(self, tmp_path: Path):
        h = AgentApprovedHandler(agent_id="<script>alert('xss')</script>")
        h.end_session()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out, organisation='Corp "& Sons"')

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html or "alert" not in html
        assert "&amp;" in html


# ── Edge Cases ──────────────────────────────────────────────────


class TestExporterEdgeCases:
    def test_empty_events(self, tmp_path: Path):
        out = tmp_path / "packet"
        generate_evidence_packet([], out, organisation="Empty Corp")

        with open(out / "evidence.json", "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == 0
        assert doc["compliance"]["overall_score"] == 0

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "0%" in html

    def test_no_public_key(self, tmp_path: Path):
        h = _full_session_handler()
        out = tmp_path / "packet"
        generate_evidence_packet(h.get_events(), out, public_key_hex="")

        with open(out / "integrity.json", "r") as f:
            doc = json.load(f)
        assert doc["chain_valid"] is True  # Chain still valid, just no sig check
        assert doc["public_key"] == ""

    def test_packet_from_persisted_session(self, tmp_path: Path):
        """Generate packet from events loaded from disk — proves full pipeline."""
        data_dir = tmp_path / "data"
        h = _full_session_handler(data_dir=data_dir)

        from agentapproved import load_session_file
        loaded = load_session_file(data_dir / "sessions" / f"{h.session_id}.json")

        out = tmp_path / "packet"
        generate_evidence_packet(
            loaded, out,
            organisation="From Disk Corp",
            public_key_hex=get_public_key_hex(h._public_key),
        )

        with open(out / "evidence.json", "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == len(h.get_events())
        assert doc["compliance"]["overall_score"] == 100
