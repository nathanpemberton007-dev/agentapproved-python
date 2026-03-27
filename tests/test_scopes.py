"""Tests for scope-specific helper methods (Singapore MGF + Integrity Oath)."""

import json

import pytest

from agentapproved import AgentApprovedHandler
from agentapproved.hasher import compute_event_hash


# ── Singapore MGF Dimension 1: Accountability ─────────────────


class TestSGPAccountability:
    def test_record_config(self):
        handler = AgentApprovedHandler(agent_id="sgp-test")
        evt = handler.record_config(config_name="max_tokens", details="Set to 4096")
        assert evt.action_type == "config"
        assert evt.action_name == "max_tokens"
        body = json.loads(evt.input_data)
        assert body["config_name"] == "max_tokens"
        assert body["details"] == "Set to 4096"
        assert evt.actor_type == "agent"

    def test_record_permission(self):
        handler = AgentApprovedHandler(agent_id="sgp-test")
        evt = handler.record_permission(
            permission="read_database",
            scope="customer_data",
            details="Read-only access to customer table",
        )
        assert evt.action_type == "permission"
        assert evt.action_name == "scope_customer_data"
        body = json.loads(evt.input_data)
        assert body["permission"] == "read_database"
        assert body["scope"] == "customer_data"
        assert body["details"] == "Read-only access to customer table"

    def test_record_permission_no_details(self):
        handler = AgentApprovedHandler()
        evt = handler.record_permission(permission="write_logs", scope="audit")
        body = json.loads(evt.input_data)
        assert body["details"] == ""

    def test_record_approval(self):
        handler = AgentApprovedHandler()
        evt = handler.record_approval(
            reviewer_id="manager-bob",
            decision="approved",
            reason="Looks good",
            related_event_id="evt-123",
        )
        assert evt.action_type == "approval"
        assert evt.action_name == "approval_approved"
        assert evt.actor_type == "human"
        assert evt.actor_id == "manager-bob"
        body = json.loads(evt.input_data)
        assert body["decision"] == "approved"
        assert body["reason"] == "Looks good"
        assert body["related_event"] == "evt-123"

    def test_record_approval_minimal(self):
        handler = AgentApprovedHandler()
        evt = handler.record_approval(reviewer_id="alice", decision="rejected")
        body = json.loads(evt.input_data)
        assert body["reason"] == ""
        assert body["related_event"] is None

    def test_record_review(self):
        handler = AgentApprovedHandler()
        evt = handler.record_review(
            reviewer_id="auditor-jane",
            subject="output_quality",
            outcome="pass",
            notes="All outputs within acceptable range",
        )
        assert evt.action_type == "review"
        assert evt.action_name == "review_output_quality"
        assert evt.actor_type == "human"
        assert evt.actor_id == "auditor-jane"
        body = json.loads(evt.input_data)
        assert body["subject"] == "output_quality"
        assert body["outcome"] == "pass"
        assert body["notes"] == "All outputs within acceptable range"

    def test_record_documentation_with_author(self):
        handler = AgentApprovedHandler()
        evt = handler.record_documentation(
            doc_type="capability_statement",
            content="This agent can process invoices",
            author_id="tech-lead-sam",
        )
        assert evt.action_type == "documentation"
        assert evt.action_name == "capability_statement"
        assert evt.actor_type == "human"
        assert evt.actor_id == "tech-lead-sam"
        body = json.loads(evt.input_data)
        assert body["doc_type"] == "capability_statement"
        assert body["content"] == "This agent can process invoices"

    def test_record_documentation_without_author(self):
        handler = AgentApprovedHandler(actor_id="system-agent")
        evt = handler.record_documentation(
            doc_type="limitation_notice",
            content="Cannot handle images",
        )
        assert evt.action_type == "documentation"
        assert evt.actor_type == "agent"
        assert evt.actor_id == "system-agent"


# ── Singapore MGF Dimension 3: Monitoring ─────────────────────


class TestSGPMonitoring:
    def test_record_test(self):
        handler = AgentApprovedHandler()
        evt = handler.record_test(
            test_name="accuracy_benchmark",
            result="pass",
            details="95.2% accuracy on validation set",
        )
        assert evt.action_type == "test"
        assert evt.action_name == "accuracy_benchmark"
        body = json.loads(evt.input_data)
        assert body["test_name"] == "accuracy_benchmark"
        assert body["result"] == "pass"
        assert body["details"] == "95.2% accuracy on validation set"

    def test_record_test_minimal(self):
        handler = AgentApprovedHandler()
        evt = handler.record_test(test_name="smoke_test", result="fail")
        body = json.loads(evt.input_data)
        assert body["details"] == ""

    def test_record_evaluation(self):
        handler = AgentApprovedHandler()
        evt = handler.record_evaluation(
            eval_name="bias_score",
            score=0.03,
            details="Gender bias evaluation on test corpus",
        )
        assert evt.action_type == "evaluation"
        assert evt.action_name == "bias_score"
        body = json.loads(evt.input_data)
        assert body["eval_name"] == "bias_score"
        assert body["score"] == 0.03
        assert body["details"] == "Gender bias evaluation on test corpus"

    def test_record_evaluation_zero_score(self):
        handler = AgentApprovedHandler()
        evt = handler.record_evaluation(eval_name="toxicity", score=0.0)
        body = json.loads(evt.input_data)
        assert body["score"] == 0.0

    def test_record_monitor(self):
        handler = AgentApprovedHandler()
        evt = handler.record_monitor(
            metric_name="response_latency_p99",
            value="230ms",
            status="ok",
        )
        assert evt.action_type == "monitor"
        assert evt.action_name == "response_latency_p99"
        body = json.loads(evt.input_data)
        assert body["metric_name"] == "response_latency_p99"
        assert body["value"] == "230ms"
        assert body["status"] == "ok"

    def test_record_monitor_warning_status(self):
        handler = AgentApprovedHandler()
        evt = handler.record_monitor(
            metric_name="error_rate",
            value="5.1%",
            status="warning",
        )
        body = json.loads(evt.input_data)
        assert body["status"] == "warning"

    def test_record_monitor_default_status(self):
        handler = AgentApprovedHandler()
        evt = handler.record_monitor(metric_name="cpu", value="42%")
        body = json.loads(evt.input_data)
        assert body["status"] == "ok"

    def test_record_health_check(self):
        handler = AgentApprovedHandler()
        evt = handler.record_health_check(
            service="database",
            status="healthy",
            details="Connection pool at 12/100",
        )
        assert evt.action_type == "health_check"
        assert evt.action_name == "database"
        body = json.loads(evt.input_data)
        assert body["service"] == "database"
        assert body["status"] == "healthy"
        assert body["details"] == "Connection pool at 12/100"

    def test_record_health_check_minimal(self):
        handler = AgentApprovedHandler()
        evt = handler.record_health_check(service="redis", status="down")
        body = json.loads(evt.input_data)
        assert body["details"] == ""

    def test_record_heartbeat(self):
        handler = AgentApprovedHandler()
        evt = handler.record_heartbeat()
        assert evt.action_type == "heartbeat"
        assert evt.action_name == "heartbeat"
        body = json.loads(evt.input_data)
        assert body["status"] == "alive"
        assert evt.actor_type == "agent"


# ── Singapore MGF Dimension 4: Transparency ───────────────────


class TestSGPTransparency:
    def test_record_disclosure(self):
        handler = AgentApprovedHandler()
        evt = handler.record_disclosure(
            disclosure_type="capability_limitation",
            content="Cannot process images or audio",
        )
        assert evt.action_type == "disclosure"
        assert evt.action_name == "capability_limitation"
        body = json.loads(evt.input_data)
        assert body["disclosure_type"] == "capability_limitation"
        assert body["content"] == "Cannot process images or audio"


# ── Integrity Oath Helpers ────────────────────────────────────


class TestIntegrityOath:
    def test_record_error(self):
        handler = AgentApprovedHandler()
        evt = handler.record_error(
            error_type="timeout",
            message="LLM call timed out after 30s",
            details="Model: gpt-4o, attempt 3 of 3",
        )
        assert evt.action_type == "error"
        assert evt.action_name == "timeout"
        body = json.loads(evt.input_data)
        assert body["error_type"] == "timeout"
        assert body["message"] == "LLM call timed out after 30s"
        assert body["details"] == "Model: gpt-4o, attempt 3 of 3"
        assert evt.actor_type == "agent"

    def test_record_error_minimal(self):
        handler = AgentApprovedHandler()
        evt = handler.record_error(error_type="parse_error", message="Invalid JSON")
        body = json.loads(evt.input_data)
        assert body["details"] == ""

    def test_record_tool_error(self):
        handler = AgentApprovedHandler()
        evt = handler.record_tool_error(
            tool_name="web_search",
            error="HTTP 429 Too Many Requests",
            details="Rate limited by provider",
        )
        assert evt.action_type == "tool_call_error"
        assert evt.action_name == "web_search"
        body = json.loads(evt.input_data)
        assert body["tool_name"] == "web_search"
        assert body["error"] == "HTTP 429 Too Many Requests"
        assert body["details"] == "Rate limited by provider"

    def test_record_tool_error_minimal(self):
        handler = AgentApprovedHandler()
        evt = handler.record_tool_error(tool_name="calculator", error="Division by zero")
        body = json.loads(evt.input_data)
        assert body["details"] == ""

    def test_record_escalation(self):
        handler = AgentApprovedHandler()
        evt = handler.record_escalation(
            reviewer_id="ops-team",
            reason="Confidence below threshold for financial decision",
            severity="high",
        )
        assert evt.action_type == "human_escalation"
        assert evt.action_name == "escalation_high"
        assert evt.actor_type == "human"
        assert evt.actor_id == "ops-team"
        body = json.loads(evt.input_data)
        assert body["reason"] == "Confidence below threshold for financial decision"
        assert body["severity"] == "high"

    def test_record_escalation_default_severity(self):
        handler = AgentApprovedHandler()
        evt = handler.record_escalation(reviewer_id="admin", reason="Unusual pattern detected")
        body = json.loads(evt.input_data)
        assert body["severity"] == "normal"
        assert evt.action_name == "escalation_normal"


# ── Human Actor Verification ──────────────────────────────────


class TestHumanActorMethods:
    """Verify all human-actor methods correctly set actor_type='human'."""

    def test_all_human_methods_set_actor_type(self):
        handler = AgentApprovedHandler()
        human_events = [
            handler.record_approval(reviewer_id="r1", decision="ok"),
            handler.record_review(reviewer_id="r2", subject="s", outcome="pass"),
            handler.record_documentation(doc_type="d", content="c", author_id="r3"),
            handler.record_escalation(reviewer_id="r4", reason="test"),
            handler.record_oversight(reviewer_id="r5", decision="approved"),
        ]
        for evt in human_events:
            assert evt.actor_type == "human", (
                f"{evt.action_type} should have actor_type='human', got '{evt.actor_type}'"
            )

    def test_non_human_methods_set_agent_actor(self):
        handler = AgentApprovedHandler()
        agent_events = [
            handler.record_config(config_name="c", details="d"),
            handler.record_permission(permission="p", scope="s"),
            handler.record_documentation(doc_type="d", content="c"),  # no author_id
            handler.record_test(test_name="t", result="pass"),
            handler.record_evaluation(eval_name="e", score=1.0),
            handler.record_monitor(metric_name="m", value="v"),
            handler.record_health_check(service="s", status="ok"),
            handler.record_heartbeat(),
            handler.record_disclosure(disclosure_type="d", content="c"),
            handler.record_error(error_type="e", message="m"),
            handler.record_tool_error(tool_name="t", error="e"),
        ]
        for evt in agent_events:
            assert evt.actor_type == "agent", (
                f"{evt.action_type} should have actor_type='agent', got '{evt.actor_type}'"
            )


# ── Data Serialization ────────────────────────────────────────


class TestDataSerialization:
    """Verify all helper methods produce valid JSON in input_data."""

    def test_all_helpers_produce_valid_json(self):
        handler = AgentApprovedHandler()
        events = [
            handler.record_config(config_name="c", details="d"),
            handler.record_permission(permission="p", scope="s", details="d"),
            handler.record_approval(reviewer_id="r", decision="ok", reason="r"),
            handler.record_review(reviewer_id="r", subject="s", outcome="o", notes="n"),
            handler.record_documentation(doc_type="d", content="c"),
            handler.record_test(test_name="t", result="pass", details="d"),
            handler.record_evaluation(eval_name="e", score=0.95, details="d"),
            handler.record_monitor(metric_name="m", value="v", status="ok"),
            handler.record_health_check(service="s", status="ok", details="d"),
            handler.record_heartbeat(),
            handler.record_disclosure(disclosure_type="d", content="c"),
            handler.record_error(error_type="e", message="m", details="d"),
            handler.record_tool_error(tool_name="t", error="e", details="d"),
            handler.record_escalation(reviewer_id="r", reason="r", severity="high"),
        ]
        for evt in events:
            parsed = json.loads(evt.input_data)
            assert isinstance(parsed, dict), (
                f"{evt.action_type} input_data should parse to dict"
            )

    def test_unicode_in_data(self):
        handler = AgentApprovedHandler()
        evt = handler.record_config(config_name="model", details="Using GPT-4o \u2014 latest version")
        body = json.loads(evt.input_data)
        assert "\u2014" in body["details"]


# ── Hash Chain Integrity With New Methods ─────────────────────


class TestChainIntegrityWithScopes:
    def test_chain_valid_after_sgp_events(self):
        handler = AgentApprovedHandler(agent_id="chain-sgp")
        handler.record_config(config_name="model", details="gpt-4o")
        handler.record_permission(permission="read", scope="db")
        handler.record_approval(reviewer_id="boss", decision="go")
        handler.record_review(reviewer_id="qa", subject="output", outcome="pass")
        handler.record_documentation(doc_type="spec", content="Agent processes invoices")
        handler.record_test(test_name="unit", result="pass")
        handler.record_evaluation(eval_name="accuracy", score=0.98)
        handler.record_monitor(metric_name="latency", value="100ms")
        handler.record_health_check(service="api", status="healthy")
        handler.record_heartbeat()
        handler.record_disclosure(disclosure_type="limitation", content="No image support")
        handler.end_session()

        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 13  # 1 session_start + 11 SGP events + 1 session_end

    def test_chain_valid_after_integrity_events(self):
        handler = AgentApprovedHandler(agent_id="chain-integrity")
        handler.record_error(error_type="timeout", message="LLM timeout")
        handler.record_tool_error(tool_name="search", error="404")
        handler.record_escalation(reviewer_id="admin", reason="Low confidence")
        handler.end_session()

        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 5  # 1 session_start + 3 integrity events + 1 session_end

    def test_chain_valid_mixed_scopes(self):
        """Use helpers from all scopes in one session and verify chain."""
        handler = AgentApprovedHandler(agent_id="chain-mixed")

        # EU AI Act (existing)
        handler.record_oversight(reviewer_id="auditor", decision="approved", reason="Good")

        # Singapore MGF
        handler.record_config(config_name="max_tokens", details="4096")
        handler.record_permission(permission="read", scope="customer_data")
        handler.record_approval(reviewer_id="mgr", decision="approved")
        handler.record_review(reviewer_id="qa", subject="quality", outcome="pass")
        handler.record_documentation(doc_type="spec", content="Handles invoices", author_id="eng")
        handler.record_test(test_name="integration", result="pass")
        handler.record_evaluation(eval_name="fairness", score=0.99)
        handler.record_monitor(metric_name="error_rate", value="0.1%")
        handler.record_health_check(service="llm_provider", status="healthy")
        handler.record_heartbeat()
        handler.record_disclosure(disclosure_type="ai_identity", content="I am an AI assistant")

        # Integrity Oath
        handler.record_error(error_type="parse", message="Bad JSON from tool")
        handler.record_tool_error(tool_name="calculator", error="Division by zero")
        handler.record_escalation(reviewer_id="human-ops", reason="Edge case", severity="high")

        handler.end_session()

        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 17  # 1 start + 1 oversight + 11 SGP + 3 integrity + 1 end

    def test_chain_links_correctly_with_new_methods(self):
        handler = AgentApprovedHandler()
        handler.record_config(config_name="a", details="b")
        handler.record_error(error_type="x", message="y")
        handler.record_heartbeat()
        handler.end_session()

        for i in range(1, len(handler.events)):
            assert handler.events[i].previous_hash == handler.events[i - 1].event_hash

    def test_sequence_numbers_monotonic_with_new_methods(self):
        handler = AgentApprovedHandler()
        handler.record_config(config_name="a", details="b")
        handler.record_approval(reviewer_id="r", decision="ok")
        handler.record_error(error_type="e", message="m")
        handler.end_session()

        sequences = [e.sequence_number for e in handler.events]
        assert sequences == list(range(len(sequences)))

    def test_tamper_detection_with_new_methods(self):
        handler = AgentApprovedHandler()
        handler.record_config(config_name="model", details="gpt-4o")
        handler.record_heartbeat()
        handler.end_session()

        # Tamper with the config event
        handler.events[1].input_data = '{"config_name": "TAMPERED", "details": "evil"}'
        valid, break_at = handler.verify_chain()
        assert valid is False
        assert break_at == 1


# ── Full Compliance Scenario ──────────────────────────────────


class TestFullComplianceScenario:
    """Simulate a realistic agent session using all scope helpers."""

    def test_full_session_lifecycle(self):
        handler = AgentApprovedHandler(agent_id="invoice-processor", actor_id="system")

        # --- Pre-deployment (SGP MGF Dim 1 & 3) ---
        handler.record_config(config_name="model", details="claude-3.5-sonnet")
        handler.record_config(config_name="temperature", details="0.0 for deterministic output")
        handler.record_permission(permission="read_invoices", scope="finance", details="Read-only")
        handler.record_permission(permission="write_results", scope="output_bucket")
        handler.record_documentation(
            doc_type="capability_statement",
            content="Extracts line items from PDF invoices",
            author_id="eng-lead",
        )
        handler.record_documentation(
            doc_type="limitation_notice",
            content="Cannot handle handwritten invoices",
        )
        handler.record_test(test_name="accuracy_benchmark", result="pass", details="97.3%")
        handler.record_test(test_name="edge_case_suite", result="pass", details="All 42 cases")
        handler.record_evaluation(eval_name="hallucination_rate", score=0.02)

        # --- Human approval to deploy (SGP MGF Dim 2) ---
        handler.record_approval(
            reviewer_id="tech-lead-sarah",
            decision="approved",
            reason="Tests pass, accuracy above threshold",
        )

        # --- Transparency (SGP MGF Dim 4) ---
        handler.record_disclosure(
            disclosure_type="ai_identity",
            content="This system uses AI to process invoices",
        )
        handler.record_disclosure(
            disclosure_type="limitation",
            content="Handwritten invoices require manual review",
        )

        # --- Runtime monitoring (SGP MGF Dim 3) ---
        handler.record_heartbeat()
        handler.record_monitor(metric_name="invoices_processed", value="150")
        handler.record_monitor(metric_name="avg_latency", value="1.2s", status="ok")
        handler.record_health_check(service="pdf_parser", status="healthy")
        handler.record_health_check(service="llm_api", status="healthy")

        # --- Error handling (Integrity Oath) ---
        handler.record_error(
            error_type="parse_failure",
            message="Could not extract table from page 3",
            details="Invoice #INV-2024-5501, falling back to OCR",
        )
        handler.record_tool_error(
            tool_name="pdf_table_extractor",
            error="No table found on page",
            details="Tried camelot and tabula, both failed",
        )

        # --- Escalation (Integrity Oath) ---
        handler.record_escalation(
            reviewer_id="finance-ops",
            reason="Invoice total mismatch: extracted $5,200 vs stated $52,000",
            severity="high",
        )

        # --- Human review (SGP MGF Dim 2 + EU AI Act) ---
        handler.record_review(
            reviewer_id="finance-ops",
            subject="flagged_invoice",
            outcome="corrected",
            notes="OCR misread decimal point. Manual correction applied.",
        )
        handler.record_oversight(
            reviewer_id="finance-ops",
            decision="approved",
            reason="Batch of 150 invoices reviewed, 1 correction made",
        )

        # --- End session ---
        handler.end_session()

        # Verify everything
        valid, count = handler.verify_chain()
        assert valid is True
        assert count == len(handler.events)

        # Verify we have events from all scopes
        action_types = {e.action_type for e in handler.events}
        assert "config" in action_types
        assert "permission" in action_types
        assert "approval" in action_types
        assert "review" in action_types
        assert "documentation" in action_types
        assert "test" in action_types
        assert "evaluation" in action_types
        assert "monitor" in action_types
        assert "health_check" in action_types
        assert "heartbeat" in action_types
        assert "disclosure" in action_types
        assert "error" in action_types
        assert "tool_call_error" in action_types
        assert "human_escalation" in action_types
        assert "human_oversight" in action_types

        # Verify human actors are correctly tagged
        human_events = [e for e in handler.events if e.actor_type == "human"]
        assert len(human_events) == 5  # approval, review, documentation(author), escalation, oversight

        # Verify chain structure
        assert handler.events[0].previous_hash == "GENESIS"
        for i in range(1, len(handler.events)):
            assert handler.events[i].previous_hash == handler.events[i - 1].event_hash
        sequences = [e.sequence_number for e in handler.events]
        assert sequences == list(range(len(sequences)))
