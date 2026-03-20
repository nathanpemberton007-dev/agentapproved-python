"""Tests for AgentApprovedHandler — proves event capture and chain integrity."""

import json
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from agentapproved import AgentApprovedHandler
from agentapproved.hasher import compute_event_hash, hash_data


# ── Schema & Hasher ─────────────────────────────────────────────


class TestHasher:
    def test_hash_data_deterministic(self):
        assert hash_data("hello") == hash_data("hello")

    def test_hash_data_different_inputs(self):
        assert hash_data("hello") != hash_data("world")

    def test_hash_data_returns_64_char_hex(self):
        h = hash_data("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_event_hash_deterministic(self):
        d = {"a": 1, "b": 2, "previous_hash": "GENESIS"}
        assert compute_event_hash(d) == compute_event_hash(d)

    def test_compute_event_hash_changes_with_data(self):
        d1 = {"a": 1, "previous_hash": "GENESIS"}
        d2 = {"a": 2, "previous_hash": "GENESIS"}
        assert compute_event_hash(d1) != compute_event_hash(d2)

    def test_compute_event_hash_changes_with_previous(self):
        d1 = {"a": 1, "previous_hash": "GENESIS"}
        d2 = {"a": 1, "previous_hash": "abc123"}
        assert compute_event_hash(d1) != compute_event_hash(d2)


# ── Handler Lifecycle ───────────────────────────────────────────


class TestHandlerLifecycle:
    def test_session_start_on_init(self):
        handler = AgentApprovedHandler(agent_id="test-agent", actor_id="user-1")
        assert len(handler.events) == 1
        first = handler.events[0]
        assert first.action_type == "session_start"
        assert first.actor_id == "user-1"
        assert first.previous_hash == "GENESIS"
        assert first.sequence_number == 0
        assert first.event_hash != ""

    def test_session_end(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        assert handler.events[-1].action_type == "session_end"
        body = json.loads(handler.events[-1].input_data)
        assert body["event_count"] == 1

    def test_session_id_consistent(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        ids = {e.session_id for e in handler.events}
        assert len(ids) == 1


# ── LLM Events ─────────────────────────────────────────────────


class TestLLMCapture:
    def test_llm_start_end(self):
        handler = AgentApprovedHandler(agent_id="test")
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "FakeLLM", "kwargs": {"model_name": "gpt-4o", "temperature": 0.7}},
            prompts=["What is compliance?"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=LLMResult(
                generations=[[Generation(text="It means following rules.")]],
                llm_output={"token_usage": {"total_tokens": 15, "prompt_tokens": 5, "completion_tokens": 10}},
            ),
            run_id=run_id,
        )

        assert len(handler.events) == 3
        start_evt = handler.events[1]
        end_evt = handler.events[2]

        assert start_evt.action_type == "llm_call_start"
        assert start_evt.model_id == "gpt-4o"
        assert start_evt.model_params["temperature"] == 0.7
        assert "compliance" in start_evt.input_data

        assert end_evt.action_type == "llm_call_end"
        assert end_evt.output_data == "It means following rules."
        assert end_evt.model_params["total_tokens"] == 15
        assert "latency_ms" in end_evt.model_params

    def test_chat_model_start(self):
        handler = AgentApprovedHandler()
        run_id = uuid4()

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI", "kwargs": {"model": "gpt-4o-mini"}},
            messages=[[HumanMessage(content="Hello"), AIMessage(content="Hi")]],
            run_id=run_id,
        )

        evt = handler.events[1]
        assert evt.action_type == "llm_call_start"
        assert evt.model_id == "gpt-4o-mini"
        msgs = json.loads(evt.input_data)
        assert len(msgs) == 2
        assert msgs[0]["type"] == "human"
        assert msgs[1]["type"] == "ai"

    def test_chat_generation_output(self):
        handler = AgentApprovedHandler()
        run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "Chat", "kwargs": {}},
            prompts=["hi"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=LLMResult(
                generations=[[ChatGeneration(message=AIMessage(content="Hello back"))]],
                llm_output=None,
            ),
            run_id=run_id,
        )
        assert handler.events[-1].output_data == "Hello back"


# ── Tool Events ─────────────────────────────────────────────────


class TestToolCapture:
    def test_tool_start_end(self):
        handler = AgentApprovedHandler()
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "DuckDuckGoSearch"},
            input_str="EU AI Act requirements",
            run_id=run_id,
        )
        handler.on_tool_end(output="The EU AI Act requires...", run_id=run_id)

        start_evt = handler.events[1]
        end_evt = handler.events[2]

        assert start_evt.action_type == "tool_call_start"
        assert start_evt.action_name == "DuckDuckGoSearch"
        assert start_evt.input_data == "EU AI Act requirements"

        assert end_evt.action_type == "tool_call_end"
        assert end_evt.output_data == "The EU AI Act requires..."

    def test_tool_none_output(self):
        handler = AgentApprovedHandler()
        handler.on_tool_end(output=None, run_id=uuid4())
        assert handler.events[-1].output_data == ""


# ── Retriever Events ────────────────────────────────────────────


class TestRetrieverCapture:
    def test_retriever_start_end(self):
        handler = AgentApprovedHandler()
        run_id = uuid4()

        handler.on_retriever_start(
            serialized={"name": "VectorStoreRetriever"},
            query="return policy",
            run_id=run_id,
        )
        handler.on_retriever_end(
            documents=[
                Document(page_content="30 day return policy applies.", metadata={"source": "policy.pdf"}),
                Document(page_content="Warranty covers defects.", metadata={"source": "warranty.pdf"}),
            ],
            run_id=run_id,
        )

        start_evt = handler.events[1]
        end_evt = handler.events[2]

        assert start_evt.action_type == "retrieval_start"
        assert start_evt.input_data == "return policy"

        assert end_evt.action_type == "retrieval_end"
        assert end_evt.retrieval_sources == ["policy.pdf", "warranty.pdf"]
        output = json.loads(end_evt.output_data)
        assert len(output) == 2
        assert "30 day" in output[0]["content_preview"]


# ── Agent Decision Events ──────────────────────────────────────


class TestAgentCapture:
    def test_agent_action(self):
        handler = AgentApprovedHandler()
        handler.on_agent_action(
            action=AgentAction(tool="search", tool_input="query", log="Thought: I should search"),
            run_id=uuid4(),
        )
        evt = handler.events[1]
        assert evt.action_type == "agent_decision"
        assert evt.action_name == "search"
        body = json.loads(evt.input_data)
        assert body["tool"] == "search"
        assert "Thought" in body["log"]

    def test_agent_finish(self):
        handler = AgentApprovedHandler()
        handler.on_agent_finish(
            finish=AgentFinish(return_values={"output": "Done."}, log="Final answer"),
            run_id=uuid4(),
        )
        evt = handler.events[1]
        assert evt.action_type == "agent_finish"
        body = json.loads(evt.output_data)
        assert body["output"] == "Done."


# ── Human Oversight ─────────────────────────────────────────────


class TestOversight:
    def test_record_oversight(self):
        handler = AgentApprovedHandler()
        handler.record_oversight(
            reviewer_id="auditor-jane",
            decision="approved",
            reason="Output is accurate",
        )
        evt = handler.events[1]
        assert evt.action_type == "human_oversight"
        assert evt.actor_type == "human"
        assert evt.actor_id == "auditor-jane"
        assert evt.action_name == "oversight_approved"
        body = json.loads(evt.input_data)
        assert body["decision"] == "approved"
        assert body["reason"] == "Output is accurate"


# ── Hash Chain Integrity ────────────────────────────────────────


class TestChainIntegrity:
    def test_chain_valid_after_multiple_events(self):
        handler = AgentApprovedHandler(agent_id="chain-test")
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "LLM", "kwargs": {}},
            prompts=["hello"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=LLMResult(generations=[[Generation(text="world")]], llm_output=None),
            run_id=run_id,
        )
        handler.on_tool_start(
            serialized={"name": "MyTool"},
            input_str="test input",
            run_id=uuid4(),
        )
        handler.record_oversight(reviewer_id="user", decision="approved")
        handler.end_session()

        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 6

    def test_chain_links_correctly(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        for i in range(1, len(handler.events)):
            assert handler.events[i].previous_hash == handler.events[i - 1].event_hash

    def test_first_event_links_to_genesis(self):
        handler = AgentApprovedHandler()
        assert handler.events[0].previous_hash == "GENESIS"

    def test_sequence_numbers_monotonic(self):
        handler = AgentApprovedHandler()
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()
        sequences = [e.sequence_number for e in handler.events]
        assert sequences == [0, 1, 2]

    def test_hash_changes_if_data_tampered(self):
        handler = AgentApprovedHandler()
        original_hash = handler.events[0].event_hash
        handler.events[0].input_data = "TAMPERED"
        computed = compute_event_hash(handler.events[0].to_hashable_dict())
        assert computed != original_hash

    def test_verify_detects_tampered_event(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        handler.events[0].input_data = "TAMPERED"
        valid, break_at = handler.verify_chain()
        assert valid is False
        assert break_at == 0


# ── Error Resilience ────────────────────────────────────────────


class TestErrorResilience:
    def test_llm_start_with_garbage_does_not_raise(self):
        handler = AgentApprovedHandler()
        count_before = len(handler.events)
        handler.on_llm_start(serialized=None, prompts=None, run_id=uuid4())
        assert len(handler.events) >= count_before

    def test_tool_end_with_complex_object(self):
        handler = AgentApprovedHandler()
        handler.on_tool_end(output={"nested": [1, 2, 3]}, run_id=uuid4())
        assert handler.events[-1].action_type == "tool_call_end"

    def test_retriever_end_with_empty_docs(self):
        handler = AgentApprovedHandler()
        handler.on_retriever_end(documents=[], run_id=uuid4())
        assert handler.events[-1].action_type == "retrieval_end"

    def test_chain_still_valid_after_swallowed_error(self):
        handler = AgentApprovedHandler()
        handler.on_llm_start(serialized=None, prompts=None, run_id=uuid4())
        handler.end_session()
        valid, _ = handler.verify_chain()
        assert valid is True


# ── Parent Event Linking ────────────────────────────────────────


class TestParentLinking:
    def test_child_event_links_to_parent(self):
        handler = AgentApprovedHandler()
        parent_run = uuid4()
        child_run = uuid4()

        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "test"},
            run_id=parent_run,
        )
        handler.on_llm_start(
            serialized={"name": "LLM", "kwargs": {}},
            prompts=["test"],
            run_id=child_run,
            parent_run_id=parent_run,
        )

        chain_evt = handler.events[1]
        llm_evt = handler.events[2]
        assert llm_evt.parent_event_id == chain_evt.event_id
