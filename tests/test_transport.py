"""Tests for local filesystem transport — proves events survive write/read cycle."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.outputs import Generation, LLMResult

from agentapproved import (
    AgentApprovedHandler,
    EvidenceEvent,
    load_session_file,
    verify_session_file,
)
from agentapproved.transport import LocalTransport


# ── Session File Creation ───────────────────────────────────────


class TestSessionPersistence:
    def test_handler_creates_session_file_on_init(self, tmp_path: Path):
        handler = AgentApprovedHandler(
            agent_id="test-agent", data_dir=tmp_path
        )
        sessions_dir = tmp_path / "sessions"
        files = list(sessions_dir.glob("*.json"))
        assert len(files) == 1
        assert handler.session_id in files[0].stem

    def test_file_updates_on_every_emit(self, tmp_path: Path):
        handler = AgentApprovedHandler(agent_id="test", data_dir=tmp_path)
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == 1  # session_start only

        handler.record_oversight(reviewer_id="bob", decision="approved")

        with open(path, "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == 2  # session_start + oversight

    def test_session_end_persists(self, tmp_path: Path):
        handler = AgentApprovedHandler(agent_id="test", data_dir=tmp_path)
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == 2
        assert doc["events"][-1]["action_type"] == "session_end"

    def test_file_contains_session_metadata(self, tmp_path: Path):
        handler = AgentApprovedHandler(
            agent_id="my-agent", data_dir=tmp_path
        )
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        assert doc["session_id"] == handler.session_id
        assert doc["agent_id"] == "my-agent"
        assert doc["chain_start"] == "GENESIS"
        assert len(doc["chain_end"]) == 64  # SHA-256 hex


# ── Write/Read Roundtrip ───────────────────────────────────────


class TestRoundtrip:
    def test_events_survive_roundtrip(self, tmp_path: Path):
        handler = AgentApprovedHandler(
            agent_id="roundtrip-test", actor_id="user-1", data_dir=tmp_path
        )
        run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "LLM", "kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello world"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=LLMResult(
                generations=[[Generation(text="Hi there")]],
                llm_output=None,
            ),
            run_id=run_id,
        )
        handler.record_oversight(
            reviewer_id="auditor", decision="approved", reason="Looks good"
        )
        handler.end_session()

        # Load from disk
        loaded = load_session_file(
            tmp_path / "sessions" / f"{handler.session_id}.json"
        )

        assert len(loaded) == len(handler.events)
        for original, reloaded in zip(handler.events, loaded):
            assert original.event_id == reloaded.event_id
            assert original.action_type == reloaded.action_type
            assert original.action_name == reloaded.action_name
            assert original.input_hash == reloaded.input_hash
            assert original.input_data == reloaded.input_data
            assert original.output_data == reloaded.output_data
            assert original.model_id == reloaded.model_id
            assert original.event_hash == reloaded.event_hash
            assert original.previous_hash == reloaded.previous_hash
            assert original.sequence_number == reloaded.sequence_number

    def test_loaded_events_are_evidence_events(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.end_session()
        loaded = load_session_file(
            tmp_path / "sessions" / f"{handler.session_id}.json"
        )
        for event in loaded:
            assert isinstance(event, EvidenceEvent)


# ── Chain Verification from File ────────────────────────────────


class TestFileVerification:
    def test_verify_session_file_valid(self, tmp_path: Path):
        handler = AgentApprovedHandler(agent_id="verify-test", data_dir=tmp_path)
        handler.on_tool_start(
            serialized={"name": "SearchTool"},
            input_str="test query",
            run_id=uuid4(),
        )
        handler.on_tool_end(output="result", run_id=uuid4())
        handler.record_oversight(reviewer_id="user", decision="approved")
        handler.end_session()

        path = tmp_path / "sessions" / f"{handler.session_id}.json"
        valid, count = verify_session_file(path)
        assert valid is True
        assert count == 5  # session_start + tool_start + tool_end + oversight + session_end

    def test_verify_detects_tampered_file(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        # Tamper with the file
        with open(path, "r") as f:
            doc = json.load(f)
        doc["events"][0]["input_data"] = "TAMPERED"
        with open(path, "w") as f:
            json.dump(doc, f)

        valid, break_at = verify_session_file(path)
        assert valid is False
        assert break_at == 0

    def test_verify_detects_deleted_event(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        # Remove the middle event
        with open(path, "r") as f:
            doc = json.load(f)
        doc["events"].pop(1)  # Remove oversight event
        doc["event_count"] = len(doc["events"])
        with open(path, "w") as f:
            json.dump(doc, f)

        valid, break_at = verify_session_file(path)
        assert valid is False
        assert break_at == 1  # session_end now links to wrong previous

    def test_verify_detects_reordered_events(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        # Swap events 1 and 2
        with open(path, "r") as f:
            doc = json.load(f)
        doc["events"][1], doc["events"][2] = doc["events"][2], doc["events"][1]
        with open(path, "w") as f:
            json.dump(doc, f)

        valid, _ = verify_session_file(path)
        assert valid is False


# ── Multiple Sessions ───────────────────────────────────────────


class TestMultipleSessions:
    def test_separate_files_per_session(self, tmp_path: Path):
        h1 = AgentApprovedHandler(agent_id="agent-a", data_dir=tmp_path)
        h2 = AgentApprovedHandler(agent_id="agent-b", data_dir=tmp_path)
        h1.end_session()
        h2.end_session()

        files = list((tmp_path / "sessions").glob("*.json"))
        assert len(files) == 2
        assert h1.session_id != h2.session_id

    def test_list_sessions(self, tmp_path: Path):
        transport = LocalTransport(tmp_path)
        assert transport.list_sessions() == []

        h1 = AgentApprovedHandler(agent_id="a", data_dir=tmp_path)
        h2 = AgentApprovedHandler(agent_id="b", data_dir=tmp_path)

        sessions = transport.list_sessions()
        assert len(sessions) == 2
        assert h1.session_id in sessions
        assert h2.session_id in sessions

    def test_session_exists(self, tmp_path: Path):
        transport = LocalTransport(tmp_path)
        handler = AgentApprovedHandler(agent_id="test", data_dir=tmp_path)
        assert transport.session_exists(handler.session_id) is True
        assert transport.session_exists("nonexistent") is False

    def test_load_specific_session(self, tmp_path: Path):
        h1 = AgentApprovedHandler(agent_id="agent-a", data_dir=tmp_path)
        h1.record_oversight(reviewer_id="x", decision="ok")
        h2 = AgentApprovedHandler(agent_id="agent-b", data_dir=tmp_path)

        transport = LocalTransport(tmp_path)
        events_a = transport.load_session(h1.session_id)
        events_b = transport.load_session(h2.session_id)

        assert len(events_a) == 2  # session_start + oversight
        assert len(events_b) == 1  # session_start only
        assert events_a[0].session_id == h1.session_id
        assert events_b[0].session_id == h2.session_id


# ── Backward Compatibility ──────────────────────────────────────


class TestBackwardCompat:
    def test_handler_without_data_dir_creates_no_files(self, tmp_path: Path):
        handler = AgentApprovedHandler(agent_id="no-persist")
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()

        # No sessions dir should exist in tmp_path
        assert not (tmp_path / "sessions").exists()
        # But in-memory events still work
        assert len(handler.events) == 3
        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 3

    def test_handler_without_data_dir_has_no_transport(self):
        handler = AgentApprovedHandler()
        assert handler._transport is None


# ── Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_session_file_structure(self, tmp_path: Path):
        """Even session_start alone produces a valid file."""
        handler = AgentApprovedHandler(data_dir=tmp_path)
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        assert doc["event_count"] == 1
        assert doc["chain_start"] == "GENESIS"
        assert doc["events"][0]["action_type"] == "session_start"

    def test_large_event_count(self, tmp_path: Path):
        """50 events persist and verify correctly."""
        handler = AgentApprovedHandler(data_dir=tmp_path)
        for i in range(49):
            handler.record_oversight(
                reviewer_id=f"user-{i}", decision="approved"
            )

        path = tmp_path / "sessions" / f"{handler.session_id}.json"
        valid, count = verify_session_file(path)
        assert valid is True
        assert count == 50  # 1 session_start + 49 oversights

    def test_persist_failure_does_not_crash_handler(self, tmp_path: Path):
        """If disk write fails, handler keeps working in-memory."""
        handler = AgentApprovedHandler(data_dir=tmp_path)

        # Make the sessions dir read-only to force write failure
        sessions_dir = tmp_path / "sessions"
        session_file = sessions_dir / f"{handler.session_id}.json"
        session_file.chmod(0o444)
        sessions_dir.chmod(0o555)

        # This should not raise — persist failure is swallowed
        handler.record_oversight(reviewer_id="x", decision="ok")

        # In-memory events still captured
        assert len(handler.events) == 2
        valid, _ = handler.verify_chain()
        assert valid is True

        # Restore permissions for cleanup
        sessions_dir.chmod(0o755)
        session_file.chmod(0o644)
