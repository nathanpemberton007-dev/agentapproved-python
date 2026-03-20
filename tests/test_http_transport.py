"""Tests for HTTP transport -- batching, retry, handler integration."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from agentapproved import AgentApprovedHandler, EvidenceEvent, HttpTransport


# ── Test Helpers ───────────────────────────────────────────────


def _make_event(seq: int = 0, session_id: str = "test-session") -> EvidenceEvent:
    """Build a minimal valid EvidenceEvent for testing."""
    return EvidenceEvent(
        event_id=f"ev-{seq:04d}",
        timestamp="2026-03-20T14:00:00+00:00",
        session_id=session_id,
        parent_event_id=None,
        actor_type="agent",
        actor_id="test-agent",
        action_type="llm_call_start",
        action_name="test-model",
        input_hash="abc123",
        input_data="test input",
        output_hash=None,
        output_data=None,
        model_id="test-model",
        model_params=None,
        retrieval_sources=None,
        sequence_number=seq,
        previous_hash="GENESIS" if seq == 0 else "prev",
        event_hash="deadbeef" * 8,
        signature="sig" * 20,
    )


class MockServer:
    """Tiny HTTP server that records received batches."""

    def __init__(self, status: int = 200, response: dict | None = None):
        self.status = status
        self.response = response
        self.received: list[list[dict]] = []
        self.request_count = 0
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                parent.received.append(body)
                parent.request_count += 1

                resp = parent.response or {
                    "accepted": len(body),
                    "rejected": 0,
                }
                self.send_response(parent.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())

            def log_message(self, format, *args):
                pass  # Silence request logs

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}"

    def stop(self):
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)


# ── Batching ───────────────────────────────────────────────────


class TestBatching:
    def test_events_buffered_until_flush(self):
        """Events sit in buffer until manually flushed."""
        mock = MockServer()
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_test",
                flush_interval=999,  # won't auto-flush during test
            )
            for i in range(5):
                transport.send(_make_event(i))

            assert transport.pending == 5
            assert mock.request_count == 0

            sent = transport.flush()
            assert sent == 5
            assert transport.pending == 0
            assert mock.request_count == 1
            assert len(mock.received[0]) == 5

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_auto_flush_at_batch_size(self):
        """Buffer flushes automatically when batch_size is reached."""
        mock = MockServer()
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_test",
                batch_size=10,
                flush_interval=999,
            )
            for i in range(10):
                transport.send(_make_event(i))

            # Give the background thread time to flush
            time.sleep(0.5)

            assert mock.request_count >= 1
            total_events = sum(len(batch) for batch in mock.received)
            assert total_events == 10

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_auto_flush_on_interval(self):
        """Buffer flushes automatically after flush_interval seconds."""
        mock = MockServer()
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_test",
                batch_size=999,  # won't trigger batch flush
                flush_interval=0.3,
            )
            transport.send(_make_event(0))
            assert mock.request_count == 0

            # Wait for interval flush
            time.sleep(1.0)

            assert mock.request_count >= 1
            assert transport.total_sent == 1

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_shutdown_flushes_remaining(self):
        """shutdown() flushes any remaining buffered events."""
        mock = MockServer()
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_test",
                flush_interval=999,
            )
            for i in range(7):
                transport.send(_make_event(i))

            transport.shutdown(timeout=5)

            total = sum(len(b) for b in mock.received)
            assert total == 7
        finally:
            mock.stop()


# ── Retry ──────────────────────────────────────────────────────


class TestRetry:
    def test_retry_on_server_error(self):
        """5xx errors trigger retries, events re-queued on failure."""
        mock = MockServer(status=500)
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_test",
                max_retries=2,
                flush_interval=999,
            )
            transport.send(_make_event(0))
            sent = transport.flush()

            assert sent == 0  # All retries failed
            assert transport.total_retries == 2
            assert transport.pending == 1  # Re-queued

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_no_retry_on_client_error(self):
        """4xx errors do NOT retry -- events are dropped."""
        mock = MockServer(status=401)
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url,
                api_key="ap_bad_key",
                max_retries=3,
                flush_interval=999,
            )
            transport.send(_make_event(0))
            sent = transport.flush()

            assert sent == 1  # Dispatched (dropped, not retried)
            assert mock.request_count == 1  # Only 1 attempt
            assert transport.total_retries == 1
            assert transport.pending == 0

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_retry_on_connection_refused(self):
        """Network errors trigger retries, events re-queued."""
        transport = HttpTransport(
            endpoint="http://127.0.0.1:1",  # Nothing listening
            api_key="ap_test",
            max_retries=2,
            flush_interval=999,
        )
        transport.send(_make_event(0))
        sent = transport.flush()

        assert sent == 0
        assert transport.total_retries == 2
        assert transport.pending == 1

        transport.shutdown(timeout=2)


# ── Stats ──────────────────────────────────────────────────────


class TestStats:
    def test_total_sent_tracks_accepted(self):
        mock = MockServer(response={"accepted": 3, "rejected": 0})
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url, api_key="ap_test", flush_interval=999
            )
            for i in range(3):
                transport.send(_make_event(i))
            transport.flush()

            assert transport.total_sent == 3
            assert transport.total_failed == 0
            assert transport.flush_count == 1

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_partial_rejection_tracked(self):
        mock = MockServer(response={"accepted": 2, "rejected": 1, "errors": [{"index": 2, "error": "bad"}]})
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url, api_key="ap_test", flush_interval=999
            )
            for i in range(3):
                transport.send(_make_event(i))
            transport.flush()

            assert transport.total_sent == 2
            assert transport.total_failed == 1

            transport.shutdown(timeout=2)
        finally:
            mock.stop()


# ── Handler Integration ────────────────────────────────────────


class TestHandlerTransportSelection:
    def test_no_params_is_memory_only(self):
        """No api_key, no data_dir = in-memory only."""
        handler = AgentApprovedHandler(agent_id="test")
        assert handler._transport is None
        assert handler._http_transport is None
        assert len(handler.events) == 1  # session_start

    def test_data_dir_uses_local_transport(self, tmp_path):
        """data_dir = LocalTransport."""
        handler = AgentApprovedHandler(agent_id="test", data_dir=tmp_path)
        assert handler._transport is not None
        assert handler._http_transport is None

    def test_api_key_uses_http_transport(self):
        """api_key = HttpTransport."""
        handler = AgentApprovedHandler(
            agent_id="test",
            api_key="ap_test123",
            endpoint="http://127.0.0.1:9999",
        )
        assert handler._http_transport is not None
        assert isinstance(handler._http_transport, HttpTransport)
        assert handler._transport is None
        handler.shutdown()

    def test_api_key_and_data_dir_uses_both(self, tmp_path):
        """api_key + data_dir = both transports active."""
        handler = AgentApprovedHandler(
            agent_id="test",
            api_key="ap_test123",
            endpoint="http://127.0.0.1:9999",
            data_dir=tmp_path,
        )
        assert handler._http_transport is not None
        assert handler._transport is not None
        handler.shutdown()

    def test_handler_sends_events_via_http(self):
        """Events emitted by the handler arrive at the server."""
        mock = MockServer()
        url = mock.start()
        try:
            handler = AgentApprovedHandler(
                agent_id="http-test",
                api_key="ap_test123",
                endpoint=url,
            )
            # session_start already emitted
            handler.record_oversight(
                reviewer_id="alice", decision="approved", reason="looks good"
            )
            handler.end_session()

            # Flush and verify
            handler.shutdown()

            total = sum(len(b) for b in mock.received)
            assert total == 3  # session_start + oversight + session_end
        finally:
            mock.stop()

    def test_handler_shutdown_is_safe_without_http(self):
        """shutdown() is a no-op when no HTTP transport configured."""
        handler = AgentApprovedHandler(agent_id="test")
        handler.shutdown()  # Should not raise


# ── Buffer Limits ──────────────────────────────────────────────


class TestBufferLimits:
    def test_buffer_cap_drops_oldest(self):
        """When buffer exceeds MAX_BUFFER_SIZE, oldest events are dropped."""
        transport = HttpTransport(
            endpoint="http://127.0.0.1:1",
            api_key="ap_test",
            flush_interval=999,
        )
        # MAX_BUFFER_SIZE is 10_000 -- push 10 with a maxlen of 5 to test the deque behavior
        transport._buffer = __import__("collections").deque(maxlen=5)
        for i in range(8):
            transport.send(_make_event(i))

        assert transport.pending == 5
        # Oldest should have been dropped
        buffered_ids = [e["event_id"] for e in transport._buffer]
        assert buffered_ids[0] == "ev-0003"
        assert buffered_ids[-1] == "ev-0007"

        transport.shutdown(timeout=1)


# ── Event Payload ──────────────────────────────────────────────


class TestEventPayload:
    def test_events_sent_as_dicts(self):
        """Events are serialized as dicts (not EvidenceEvent objects)."""
        mock = MockServer()
        url = mock.start()
        try:
            transport = HttpTransport(
                endpoint=url, api_key="ap_test", flush_interval=999
            )
            transport.send(_make_event(0))
            transport.flush()

            batch = mock.received[0]
            assert isinstance(batch[0], dict)
            assert batch[0]["event_id"] == "ev-0000"
            assert batch[0]["action_type"] == "llm_call_start"

            transport.shutdown(timeout=2)
        finally:
            mock.stop()

    def test_auth_header_sent(self):
        """Bearer token is included in the Authorization header."""
        headers_seen = []

        class HeaderCapture(BaseHTTPRequestHandler):
            def do_POST(self):
                headers_seen.append(dict(self.headers))
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"accepted": 1, "rejected": 0}).encode())

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), HeaderCapture)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            transport = HttpTransport(
                endpoint=f"http://127.0.0.1:{port}",
                api_key="ap_my_secret_key",
                flush_interval=999,
            )
            transport.send(_make_event(0))
            transport.flush()

            assert len(headers_seen) == 1
            assert headers_seen[0].get("Authorization") == "Bearer ap_my_secret_key"

            transport.shutdown(timeout=2)
        finally:
            server.shutdown()
