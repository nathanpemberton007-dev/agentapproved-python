"""SHA-256 hashing and Ed25519 signing for event integrity chains."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def hash_data(data: str) -> str:
    """SHA-256 hash of a string payload."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_event_hash(hashable_dict: dict) -> str:
    """SHA-256 of an event's hashable fields (excludes event_hash and signature).

    The dict must contain previous_hash, linking this event to its
    predecessor in the chain.
    """
    canonical = json.dumps(hashable_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Ed25519 Key Management ──────────────────────────────────────


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def sign_hash(event_hash: str, private_key: Ed25519PrivateKey) -> str:
    """Sign an event hash with Ed25519. Returns hex-encoded signature."""
    sig_bytes = private_key.sign(bytes.fromhex(event_hash))
    return sig_bytes.hex()


def verify_signature(
    event_hash: str, signature: str, public_key: Ed25519PublicKey
) -> bool:
    """Verify an Ed25519 signature against an event hash."""
    try:
        public_key.verify(bytes.fromhex(signature), bytes.fromhex(event_hash))
        return True
    except Exception:
        return False


def save_keypair(
    private_key: Ed25519PrivateKey, data_dir: Path
) -> tuple[Path, Path]:
    """Save keypair to PEM files in data_dir/keys/."""
    keys_dir = data_dir / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    priv_path = keys_dir / "private.pem"
    pub_path = keys_dir / "public.pem"

    priv_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    pub_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )

    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)
    return priv_path, pub_path


def load_private_key(data_dir: Path) -> Ed25519PrivateKey | None:
    """Load private key from data_dir/keys/private.pem, or None if missing."""
    priv_path = data_dir / "keys" / "private.pem"
    if not priv_path.exists():
        return None
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    return load_pem_private_key(priv_path.read_bytes(), password=None)


def load_public_key(data_dir: Path) -> Ed25519PublicKey | None:
    """Load public key from data_dir/keys/public.pem, or None if missing."""
    pub_path = data_dir / "keys" / "public.pem"
    if not pub_path.exists():
        return None
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    return load_pem_public_key(pub_path.read_bytes())


def load_or_create_keypair(
    data_dir: Path,
) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Load existing keypair from data_dir, or generate and save a new one."""
    private_key = load_private_key(data_dir)
    if private_key is not None:
        return private_key, private_key.public_key()

    private_key, public_key = generate_keypair()
    save_keypair(private_key, data_dir)
    return private_key, public_key


def get_public_key_hex(public_key: Ed25519PublicKey) -> str:
    """Get the raw 32-byte public key as a hex string (for embedding in exports)."""
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return raw.hex()


def public_key_from_hex(hex_str: str) -> Ed25519PublicKey:
    """Reconstruct a public key from a hex string."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as Cls,
    )
    return Cls.from_public_bytes(bytes.fromhex(hex_str))
