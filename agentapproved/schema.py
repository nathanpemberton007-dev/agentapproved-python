"""EvidenceEvent — the atomic unit of compliance evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class EvidenceEvent:
    """A single captured agent action with integrity chain metadata."""

    # Identity
    event_id: str
    timestamp: str
    session_id: str
    parent_event_id: str | None

    # Actor
    actor_type: str
    actor_id: str

    # Action
    action_type: str
    action_name: str

    # Data
    input_hash: str
    input_data: str | None
    output_hash: str | None
    output_data: str | None

    # Context
    model_id: str | None
    model_params: dict | None
    retrieval_sources: list[str] | None

    # Integrity
    sequence_number: int
    previous_hash: str
    event_hash: str
    signature: str  # Ed25519 signature of event_hash (hex-encoded)

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceEvent:
        """Reconstruct an EvidenceEvent from a dict (e.g. loaded from JSON)."""
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_hashable_dict(self) -> dict:
        """All fields except event_hash and signature — input to hash computation."""
        d = asdict(self)
        del d["event_hash"]
        del d["signature"]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)
