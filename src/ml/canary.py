"""Canary routing: send a fixed fraction of traffic to the canary model version.

Why deterministic hashing on shipment_id instead of random():
1. Stickiness — the same shipment always hits the same model version, so a
   client retrying a request cannot observe two different predictions.
2. Debuggability — given a shipment_id you can tell offline which version
   served it, without needing the request logs.
3. Testability — the 90/10 split is exactly reproducible in tests.
"""
import hashlib

from src.config import settings


def route_to_canary(shipment_id: str, canary_fraction: float | None = None) -> bool:
    """True if this shipment should be served by the canary version."""
    fraction = settings.canary_fraction if canary_fraction is None else canary_fraction
    if fraction <= 0:
        return False
    # md5 used as a uniform hash, not for security
    digest = hashlib.md5(shipment_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100  # 0..99
    return bucket < fraction * 100
