"""HTTP transport -- sends evidence events to the AgentApproved server in batches.

Batching: flush every 50 events OR every 5 seconds (whichever comes first).
Background daemon thread -- never blocks the agent.
Retries with exponential backoff on server/network errors.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

from .schema import EvidenceEvent

logger = logging.getLogger("agentapproved")

BATCH_SIZE = 50
FLUSH_INTERVAL = 5.0
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0
MAX_BUFFER_SIZE = 10_000


class HttpTransport:
    """Batches EvidenceEvents and POSTs them to the AgentApproved server.

    - Buffers events, flushes every ``batch_size`` events or ``flush_interval`` seconds.
    - Flush runs on a daemon background thread -- never blocks the agent.
    - Retries failed POSTs with exponential backoff (1s, 2s, 4s).
    - If the server is unreachable, events stay in the buffer for the next flush.
    - Buffer is capped at MAX_BUFFER_SIZE; oldest events are dropped if it fills.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        batch_size: int = BATCH_SIZE,
        flush_interval: float = FLUSH_INTERVAL,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries

        self._buffer: deque[dict[str, Any]] = deque(maxlen=MAX_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_now = threading.Event()

        # Observable stats (for testing / debugging)
        self.total_sent = 0
        self.total_failed = 0
        self.total_retries = 0
        self.flush_count = 0

        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="agentapproved-http"
        )
        self._thread.start()

    # ── Public API ──────────────────────────────────────────────

    def send(self, event: EvidenceEvent) -> None:
        """Add an event to the buffer. Non-blocking."""
        with self._lock:
            self._buffer.append(event.to_dict())
            if len(self._buffer) >= self.batch_size:
                self._flush_now.set()

    def flush(self) -> int:
        """Flush the current buffer immediately. Returns count of events dispatched."""
        return self._do_flush()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the background thread and flush remaining events."""
        self._stop.set()
        self._flush_now.set()
        self._thread.join(timeout=timeout)
        # Final flush in calling thread
        self._do_flush()

    @property
    def pending(self) -> int:
        """Number of events waiting in the buffer."""
        with self._lock:
            return len(self._buffer)

    # ── Internal ────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: flush on interval or when batch_size reached."""
        while not self._stop.is_set():
            self._flush_now.wait(timeout=self.flush_interval)
            self._flush_now.clear()
            if self._stop.is_set():
                break
            self._do_flush()

    def _do_flush(self) -> int:
        """Take all buffered events and POST them. Returns dispatched count."""
        with self._lock:
            if not self._buffer:
                return 0
            batch = list(self._buffer)
            self._buffer.clear()

        sent = self._post_batch(batch)
        self.flush_count += 1

        if sent < len(batch):
            # Re-queue unsent events at the front of the buffer
            failed = batch[sent:]
            with self._lock:
                for event in reversed(failed):
                    self._buffer.appendleft(event)
        return sent

    def _post_batch(self, batch: list[dict]) -> int:
        """POST a batch to the server with retries. Returns count dispatched.

        Returns len(batch) on success (or 4xx client error -- no point retrying).
        Returns 0 when all retries are exhausted (events go back to buffer).
        """
        url = f"{self.endpoint}/v1/events"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(batch).encode("utf-8")

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    accepted = result.get("accepted", 0)
                    rejected = result.get("rejected", 0)
                    self.total_sent += accepted
                    if rejected > 0:
                        self.total_failed += rejected
                        logger.warning(
                            "agentapproved: %d events rejected: %s",
                            rejected,
                            result.get("errors"),
                        )
                    return len(batch)

            except urllib.error.HTTPError as e:
                logger.warning(
                    "agentapproved: POST failed (HTTP %d), attempt %d/%d",
                    e.code,
                    attempt + 1,
                    self.max_retries,
                )
                self.total_retries += 1
                if e.code < 500:
                    # Client error (4xx) -- don't retry
                    self.total_failed += len(batch)
                    logger.error(
                        "agentapproved: client error %d, dropping %d events",
                        e.code,
                        len(batch),
                    )
                    return len(batch)

            except (urllib.error.URLError, OSError, TimeoutError) as e:
                logger.warning(
                    "agentapproved: POST failed (%s), attempt %d/%d",
                    e,
                    attempt + 1,
                    self.max_retries,
                )
                self.total_retries += 1

            # Exponential backoff before next retry
            if attempt < self.max_retries - 1:
                backoff = RETRY_BACKOFF * (2**attempt)
                time.sleep(backoff)

        # All retries exhausted
        logger.error(
            "agentapproved: all %d retries failed, %d events re-queued",
            self.max_retries,
            len(batch),
        )
        return 0
