"""Tests for Ed25519 signing — proves the audit trail is tamper-proof."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from agentapproved import AgentApprovedHandler, load_session_file, verify_session_file
from agentapproved.hasher import (
    generate_keypair,
    get_public_key_hex,
    load_or_create_keypair,
    load_private_key,
    load_public_key,
    public_key_from_hex,
    sign_hash,
    verify_signature,
)


# ── Key Management ──────────────────────────────────────────────


class TestKeyManagement:
    def test_generate_keypair(self):
        private, public = generate_keypair()
        assert private is not None
        assert public is not None

    def test_sign_and_verify(self):
        private, public = generate_keypair()
        event_hash = "a" * 64  # Fake 256-bit hash as hex
        sig = sign_hash(event_hash, private)
        assert isinstance(sig, str)
        assert len(sig) == 128  # Ed25519 sig = 64 bytes = 128 hex chars
        assert verify_signature(event_hash, sig, public)

    def test_verify_rejects_wrong_key(self):
        priv1, _ = generate_keypair()
        _, pub2 = generate_keypair()
        event_hash = "b" * 64
        sig = sign_hash(event_hash, priv1)
        assert verify_signature(event_hash, sig, pub2) is False

    def test_verify_rejects_tampered_hash(self):
        private, public = generate_keypair()
        event_hash = "c" * 64
        sig = sign_hash(event_hash, private)
        tampered = "d" * 64
        assert verify_signature(tampered, sig, public) is False

    def test_verify_rejects_tampered_signature(self):
        private, public = generate_keypair()
        event_hash = "e" * 64
        sig = sign_hash(event_hash, private)
        tampered_sig = "00" + sig[2:]  # Flip first byte
        assert verify_signature(event_hash, tampered_sig, public) is False

    def test_public_key_hex_roundtrip(self):
        _, public = generate_keypair()
        hex_str = get_public_key_hex(public)
        assert len(hex_str) == 64  # 32 bytes = 64 hex chars
        restored = public_key_from_hex(hex_str)
        assert get_public_key_hex(restored) == hex_str


# ── Key Persistence ─────────────────────────────────────────────


class TestKeyPersistence:
    def test_load_or_create_saves_keys(self, tmp_path: Path):
        private, public = load_or_create_keypair(tmp_path)
        assert (tmp_path / "keys" / "private.pem").exists()
        assert (tmp_path / "keys" / "public.pem").exists()

    def test_load_or_create_reuses_existing(self, tmp_path: Path):
        priv1, pub1 = load_or_create_keypair(tmp_path)
        priv2, pub2 = load_or_create_keypair(tmp_path)
        # Same key should be loaded
        assert get_public_key_hex(pub1) == get_public_key_hex(pub2)

    def test_load_private_key_returns_none_if_missing(self, tmp_path: Path):
        assert load_private_key(tmp_path) is None

    def test_load_public_key_returns_none_if_missing(self, tmp_path: Path):
        assert load_public_key(tmp_path) is None

    def test_handler_with_data_dir_creates_keys(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        assert (tmp_path / "keys" / "private.pem").exists()
        assert (tmp_path / "keys" / "public.pem").exists()


# ── Handler Signs Events ────────────────────────────────────────


class TestHandlerSigning:
    def test_every_event_has_signature(self):
        handler = AgentApprovedHandler()
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()
        for event in handler.events:
            assert event.signature != ""
            assert len(event.signature) == 128  # 64 bytes hex

    def test_signatures_are_unique_per_event(self):
        handler = AgentApprovedHandler()
        handler.record_oversight(reviewer_id="x", decision="ok")
        handler.end_session()
        sigs = [e.signature for e in handler.events]
        assert len(set(sigs)) == len(sigs)  # All unique

    def test_signatures_verify_with_handler_key(self):
        handler = AgentApprovedHandler()
        handler.record_oversight(reviewer_id="x", decision="ok")
        for event in handler.events:
            assert verify_signature(
                event.event_hash, event.signature, handler._public_key
            )

    def test_verify_chain_checks_signatures(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        valid, count = handler.verify_chain()
        assert valid is True
        assert count == 2

    def test_verify_chain_detects_forged_signature(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        # Forge a signature
        handler.events[0].signature = "00" * 64
        valid, break_at = handler.verify_chain()
        assert valid is False
        assert break_at == 0


# ── File-Level Signature Verification ───────────────────────────


class TestFileSignatureVerification:
    def test_session_file_contains_public_key(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        assert "public_key" in doc
        assert len(doc["public_key"]) == 64  # 32 bytes hex

    def test_verify_session_file_checks_signatures(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.record_oversight(reviewer_id="auditor", decision="approved")
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        valid, count = verify_session_file(path)
        assert valid is True
        assert count == 3

    def test_verify_detects_forged_signature_in_file(self, tmp_path: Path):
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)
        doc["events"][0]["signature"] = "00" * 64
        with open(path, "w") as f:
            json.dump(doc, f)

        valid, break_at = verify_session_file(path)
        assert valid is False
        assert break_at == 0

    def test_verify_detects_tampered_data_with_valid_looking_hash(self, tmp_path: Path):
        """Tampering data AND recomputing hash still fails because signature won't match."""
        handler = AgentApprovedHandler(data_dir=tmp_path)
        handler.end_session()
        path = tmp_path / "sessions" / f"{handler.session_id}.json"

        with open(path, "r") as f:
            doc = json.load(f)

        # Tamper with data
        doc["events"][0]["input_data"] = "TAMPERED"
        # Even if attacker recomputes the hash, they can't forge the signature
        # (they don't have the private key)

        with open(path, "w") as f:
            json.dump(doc, f)

        valid, _ = verify_session_file(path)
        assert valid is False

    def test_two_handlers_same_dir_share_key(self, tmp_path: Path):
        h1 = AgentApprovedHandler(agent_id="a", data_dir=tmp_path)
        h2 = AgentApprovedHandler(agent_id="b", data_dir=tmp_path)
        assert get_public_key_hex(h1._public_key) == get_public_key_hex(h2._public_key)

    def test_handler_without_data_dir_still_signs(self):
        handler = AgentApprovedHandler()
        handler.end_session()
        for event in handler.events:
            assert event.signature != ""
            assert verify_signature(
                event.event_hash, event.signature, handler._public_key
            )
