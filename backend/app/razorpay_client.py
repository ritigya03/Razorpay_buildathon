"""Thin Razorpay wrapper — order creation + signature verification.

All calls are test-mode. When keys are not configured, order creation raises a
clear error and the API layer returns 503; webhook signature verification still
works if a webhook secret is set (useful for local replay of captured events).
"""
from __future__ import annotations

import hashlib
import hmac

import razorpay

from .config import settings

_client: razorpay.Client | None = None
if settings.razorpay_ready:
    _client = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def create_order(amount_paise: int, receipt: str, notes: dict | None = None) -> dict:
    if _client is None:
        raise RuntimeError("Razorpay keys not configured (set SENTINEL_RAZORPAY_KEY_ID / _SECRET)")
    return _client.order.create({
        "amount": amount_paise, "currency": "INR", "receipt": receipt,
        "notes": notes or {}, "payment_capture": 1,
    })


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    if not settings.razorpay_key_secret:
        return False
    expected = hmac.new(
        settings.razorpay_key_secret.encode(),
        f"{order_id}|{payment_id}".encode(), hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_webhook_signature(body: bytes, signature: str | None) -> bool:
    secret = settings.razorpay_webhook_secret
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
