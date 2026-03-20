"""Local filesystem transport — persists evidence events as JSON session files."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .hasher import (
    compute_event_hash,
    get_public_key_hex,
    load_public_key,
    public_key_from_hex,
    verify_signature,
)
from .schema import EvidenceEvent

logger = logging.getLogger("agentapproved")


class LocalTransport:
    """Persists evidence events to local JSON files, one per session.

    File layout:
        {data_dir}/sessions/{session_id}.json
        {data_dir}/keys/private.pem
        {data_dir}/keys/public.pem
    """

    def __init__(self, data_dir: str | Path = "./data") -> None:
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def persist(
        self, session_id: str, agent_id: str, events: list[EvidenceEvent]
    ) -> Path:
        """Write all events for a session to a JSON file. Atomic write via rename."""
        path = self._session_path(session_id)

        # Include public key so the file is self-verifiable
        pub_key = load_public_key(self.data_dir)
        pub_key_hex = get_public_key_hex(pub_key) if pub_key else ""

        doc = {
            "session_id": session_id,
            "agent_id": agent_id,
            "event_count": len(events),
            "chain_start": events[0].previous_hash if events else "GENESIS",
            "chain_end": events[-1].event_hash if events else "",
            "public_key": pub_key_hex,
            "events": [e.to_dict() for e in events],
        }

        fd, tmp_path = tempfile.mkstemp(
            dir=self.sessions_dir, suffix=".tmp", prefix=".session_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, sort_keys=True, default=str)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return path

    def load_session(self, session_id: str) -> list[EvidenceEvent]:
        """Read a session file and return its events."""
        return load_session_file(self._session_path(session_id))

    def list_sessions(self) -> list[str]:
        """Return all persisted session IDs, sorted."""
        return [p.stem for p in sorted(self.sessions_dir.glob("*.json"))]

    def session_exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"


# ── Standalone functions (no transport instance needed) ─────────


def load_session_file(path: str | Path) -> list[EvidenceEvent]:
    """Load events from a session JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return [EvidenceEvent.from_dict(e) for e in doc["events"]]


def load_session_doc(path: str | Path) -> dict:
    """Load the full session document (including metadata and public key)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_session_file(path: str | Path) -> tuple[bool, int]:
    """Verify hash chain integrity AND signatures of a persisted session file.

    Returns (is_valid, event_count). Works independently — no handler needed.
    The public key is embedded in the session file itself.
    """
    doc = load_session_doc(path)
    events = [EvidenceEvent.from_dict(e) for e in doc["events"]]
    pub_key_hex = doc.get("public_key", "")

    # Load public key for signature verification
    pub_key = None
    if pub_key_hex:
        try:
            pub_key = public_key_from_hex(pub_key_hex)
        except Exception:
            pass

    for i, event in enumerate(events):
        expected_prev = "GENESIS" if i == 0 else events[i - 1].event_hash
        if event.previous_hash != expected_prev:
            return False, i
        computed = compute_event_hash(event.to_hashable_dict())
        if computed != event.event_hash:
            return False, i
        # Verify signature if public key available
        if pub_key and event.signature:
            if not verify_signature(event.event_hash, event.signature, pub_key):
                return False, i

    return True, len(events)
