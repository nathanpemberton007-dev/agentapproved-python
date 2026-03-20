"""LangChain callback handler that captures agent events as compliance evidence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from uuid_utils import uuid7

from .hasher import (
    compute_event_hash,
    generate_keypair,
    hash_data,
    load_or_create_keypair,
    sign_hash,
)
from .schema import EvidenceEvent
from .transport import LocalTransport

logger = logging.getLogger("agentapproved")


class AgentApprovedHandler(BaseCallbackHandler):
    """Captures every LangChain agent event as a hash-chained, signed EvidenceEvent.

    Usage (in-memory only):
        handler = AgentApprovedHandler(agent_id="my-agent")

    Usage (persisted to disk with signing key):
        handler = AgentApprovedHandler(agent_id="my-agent", data_dir="./data")

    Usage (sent to AgentApproved server):
        handler = AgentApprovedHandler(
            agent_id="my-agent",
            api_key="ap_abc123",
            endpoint="https://api.agentapproved.ai",
        )
    """

    def __init__(
        self,
        agent_id: str = "default-agent",
        actor_id: str = "system",
        endpoint: str = "http://localhost:3000",
        data_dir: str | Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.actor_id = actor_id
        self.endpoint = endpoint
        self.session_id = str(uuid7())
        self.events: list[EvidenceEvent] = []
        self._sequence = 0
        self._previous_hash = "GENESIS"
        self._run_to_event: dict[str, str] = {}
        self._llm_start_times: dict[str, datetime] = {}
        self._transport: LocalTransport | None = None
        self._http_transport: "HttpTransport | None" = None

        if api_key is not None:
            from .http_transport import HttpTransport

            self._http_transport = HttpTransport(
                endpoint=endpoint, api_key=api_key
            )
        if data_dir is not None:
            self._transport = LocalTransport(data_dir)
            self._private_key, self._public_key = load_or_create_keypair(
                Path(data_dir)
            )
        else:
            self._private_key, self._public_key = generate_keypair()

        self._emit(
            action_type="session_start",
            action_name="session",
            input_data=json.dumps({"agent_id": agent_id, "actor_id": actor_id}),
        )

    # ── LLM Events ──────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            model_id = _extract_model_id(serialized)
            model_params = _extract_model_params(serialized)
            self._llm_start_times[str(run_id)] = datetime.now(timezone.utc)
            self._emit(
                action_type="llm_call_start",
                action_name=model_id,
                input_data=json.dumps(prompts, default=str),
                model_id=model_id,
                model_params=model_params,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_llm_start failed")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            model_id = _extract_model_id(serialized)
            model_params = _extract_model_params(serialized)
            msg_data = [
                {
                    "type": getattr(msg, "type", "unknown"),
                    "content": str(getattr(msg, "content", msg)),
                }
                for group in messages
                for msg in group
            ]
            self._llm_start_times[str(run_id)] = datetime.now(timezone.utc)
            self._emit(
                action_type="llm_call_start",
                action_name=model_id,
                input_data=json.dumps(msg_data, default=str),
                model_id=model_id,
                model_params=model_params,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_chat_model_start failed")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            output_text = _extract_llm_output(response)
            token_usage = _extract_token_usage(response)
            latency_ms = self._pop_latency(run_id)

            model_params: dict[str, Any] | None = None
            if token_usage or latency_ms is not None:
                model_params = {}
                if token_usage:
                    model_params.update(token_usage)
                if latency_ms is not None:
                    model_params["latency_ms"] = latency_ms

            self._emit(
                action_type="llm_call_end",
                action_name="llm",
                output_data=output_text,
                model_params=model_params,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_llm_end failed")

    # ── Tool Events ─────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            tool_name = serialized.get("name", "unknown_tool")
            self._emit(
                action_type="tool_call_start",
                action_name=tool_name,
                input_data=input_str,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_tool_start failed")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._emit(
                action_type="tool_call_end",
                action_name="tool",
                output_data=str(output) if output is not None else "",
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_tool_end failed")

    # ── Retriever Events ────────────────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = serialized.get("name", "retriever")
            self._emit(
                action_type="retrieval_start",
                action_name=name,
                input_data=query,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_retriever_start failed")

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            sources = [
                str(doc.metadata.get("source", doc.metadata.get("id", f"doc_{i}")))
                for i, doc in enumerate(documents)
            ]
            output_data = json.dumps(
                [
                    {
                        "content_preview": doc.page_content[:200],
                        "metadata": doc.metadata,
                    }
                    for doc in documents
                ],
                default=str,
            )
            self._emit(
                action_type="retrieval_end",
                action_name="retriever",
                output_data=output_data,
                retrieval_sources=sources or None,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_retriever_end failed")

    # ── Agent Decision Events ───────────────────────────────────

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            input_data = json.dumps(
                {"tool": action.tool, "tool_input": action.tool_input, "log": action.log},
                default=str,
            )
            self._emit(
                action_type="agent_decision",
                action_name=action.tool,
                input_data=input_data,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_agent_action failed")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._emit(
                action_type="agent_finish",
                action_name="agent",
                output_data=json.dumps(finish.return_values, default=str),
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_agent_finish failed")

    # ── Chain Events (NICE-to-have) ─────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = serialized.get("name") or serialized.get("id", ["chain"])[-1]
            self._emit(
                action_type="chain_start",
                action_name=str(name),
                input_data=json.dumps(inputs, default=str),
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_chain_start failed")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._emit(
                action_type="chain_end",
                action_name="chain",
                output_data=json.dumps(outputs, default=str),
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        except Exception:
            logger.exception("agentapproved: on_chain_end failed")

    # ── Human Oversight ─────────────────────────────────────────

    def record_oversight(
        self,
        reviewer_id: str,
        decision: str,
        reason: str = "",
        related_event_id: str | None = None,
    ) -> EvidenceEvent:
        """Record a human oversight event (EU AI Act Art 12(2)(d))."""
        return self._emit(
            action_type="human_oversight",
            action_name=f"oversight_{decision}",
            actor_type="human",
            actor_id=reviewer_id,
            input_data=json.dumps(
                {"decision": decision, "reason": reason, "related_event": related_event_id}
            ),
        )

    # ── Session Control ─────────────────────────────────────────

    def end_session(self) -> EvidenceEvent:
        """Emit session_end event."""
        return self._emit(
            action_type="session_end",
            action_name="session",
            input_data=json.dumps({"event_count": len(self.events)}),
        )

    # ── Accessors ───────────────────────────────────────────────

    def get_events(self) -> list[EvidenceEvent]:
        return list(self.events)

    def verify_chain(self) -> tuple[bool, int]:
        """Walk the hash chain and verify signatures. Returns (is_valid, event_count)."""
        from .hasher import verify_signature

        for i, event in enumerate(self.events):
            expected_prev = "GENESIS" if i == 0 else self.events[i - 1].event_hash
            if event.previous_hash != expected_prev:
                return False, i
            computed = compute_event_hash(event.to_hashable_dict())
            if computed != event.event_hash:
                return False, i
            if not verify_signature(event.event_hash, event.signature, self._public_key):
                return False, i
        return True, len(self.events)

    # ── Internal ────────────────────────────────────────────────

    def _emit(
        self,
        action_type: str,
        action_name: str,
        input_data: str | None = None,
        output_data: str | None = None,
        model_id: str | None = None,
        model_params: dict | None = None,
        retrieval_sources: list[str] | None = None,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        actor_type: str | None = None,
        actor_id: str | None = None,
    ) -> EvidenceEvent:
        event_id = str(uuid7())

        parent_event_id = None
        if parent_run_id:
            parent_event_id = self._run_to_event.get(str(parent_run_id))
        if run_id:
            self._run_to_event[str(run_id)] = event_id

        event = EvidenceEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            parent_event_id=parent_event_id,
            actor_type=actor_type or "agent",
            actor_id=actor_id or self.actor_id,
            action_type=action_type,
            action_name=action_name,
            input_hash=hash_data(input_data) if input_data else "",
            input_data=input_data,
            output_hash=hash_data(output_data) if output_data else None,
            output_data=output_data,
            model_id=model_id,
            model_params=model_params,
            retrieval_sources=retrieval_sources,
            sequence_number=self._sequence,
            previous_hash=self._previous_hash,
            event_hash="",
            signature="",
        )

        event.event_hash = compute_event_hash(event.to_hashable_dict())
        event.signature = sign_hash(event.event_hash, self._private_key)
        self._previous_hash = event.event_hash
        self._sequence += 1
        self.events.append(event)
        self._persist()
        return event

    def shutdown(self) -> None:
        """Flush any pending HTTP events and stop the background thread.

        Call this when the agent session is done to ensure all events are sent.
        Safe to call even if no HTTP transport is configured (no-op).
        """
        if self._http_transport is not None:
            self._http_transport.shutdown()

    def _persist(self) -> None:
        """Send event to transport(s) if configured."""
        if self._http_transport is not None:
            try:
                self._http_transport.send(self.events[-1])
            except Exception:
                logger.exception("agentapproved: http send failed")
        if self._transport is not None:
            try:
                self._transport.persist(self.session_id, self.agent_id, self.events)
            except Exception:
                logger.exception("agentapproved: persist failed")

    def _pop_latency(self, run_id: UUID) -> float | None:
        start = self._llm_start_times.pop(str(run_id), None)
        if start is None:
            return None
        return round((datetime.now(timezone.utc) - start).total_seconds() * 1000, 1)


# ── Helpers ─────────────────────────────────────────────────────


def _extract_model_id(serialized: dict[str, Any]) -> str:
    kwargs = serialized.get("kwargs", {})
    return kwargs.get("model_name") or kwargs.get("model") or serialized.get("name", "unknown")


def _extract_model_params(serialized: dict[str, Any]) -> dict | None:
    kwargs = serialized.get("kwargs", {})
    params = {
        k: v for k, v in kwargs.items()
        if k in ("temperature", "top_p", "max_tokens", "model_name", "model")
    }
    return params or None


def _extract_llm_output(response: LLMResult) -> str:
    if not response.generations:
        return ""
    first_gen = response.generations[0]
    if not first_gen:
        return ""
    gen = first_gen[0]
    if hasattr(gen, "message"):
        return str(getattr(gen.message, "content", ""))
    return getattr(gen, "text", "")


def _extract_token_usage(response: LLMResult) -> dict | None:
    if not response.llm_output:
        return None
    usage = response.llm_output.get("token_usage")
    return dict(usage) if usage else None
