"""Deterministic identifiers and seeds for persisted study records."""

import hashlib
import json

def stable_id(kind: str, *components) -> str:
    encoded = json.dumps(
        [kind, *components], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()[:20]}"


def stable_seed(*components) -> int:
    encoded = json.dumps(components, ensure_ascii=False, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


