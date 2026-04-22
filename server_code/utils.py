"""Cross-cutting helpers: API-key auth, rate limit, request logging.

Structurally identical to gift-API's utils.py — only the log schema
differs (we log citations + related_links + region filter, no urgency
triage).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

import anvil.secrets
import anvil.server
from anvil.tables import app_tables

RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_CALLS = 30  # per key per minute


# ---------------------------------------------------------------------------
# API keys


def _all_api_keys() -> dict[str, str]:
    """Return {alias: key_value} for every secret named API_KEY_*.

    Anvil has no enumerate-secrets API, so the app advertises the set of
    configured aliases in a dedicated API_KEY_ALIASES secret (comma-
    separated). Add a new key by: (1) creating secret API_KEY_<alias>,
    (2) appending the alias to API_KEY_ALIASES.
    """
    out: dict[str, str] = {}
    try:
        aliases_str = anvil.secrets.get_secret("API_KEY_ALIASES") or ""
    except Exception:
        return out
    for alias in aliases_str.split(","):
        alias = alias.strip()
        if not alias:
            continue
        try:
            out[alias] = anvil.secrets.get_secret(f"API_KEY_{alias}")
        except Exception:
            continue
    return out


def authenticate(request) -> Optional[str]:
    """Return the caller's alias if the X-API-Key header matches a
    configured key; else None.
    """
    header_key = None
    headers = getattr(request, "headers", None) or {}
    for h in ("X-API-Key", "x-api-key", "X-Api-Key"):
        if headers.get(h):
            header_key = headers.get(h)
            break
    if not header_key:
        return None

    for alias, value in _all_api_keys().items():
        if value and value == header_key:
            return alias
    return None


# ---------------------------------------------------------------------------
# Rate limit (simple per-key per-minute counter)


def check_rate_limit(alias: str) -> bool:
    """Return True if the call is allowed, False if over the limit."""
    window_start = dt.datetime.utcnow().replace(second=0, microsecond=0)
    try:
        row = app_tables.api_usage.get(key_alias=alias, window_start=window_start)
    except Exception:
        row = None
    if row is None:
        try:
            app_tables.api_usage.add_row(
                key_alias=alias, window_start=window_start, count=1
            )
        except Exception:
            pass
        return True
    try:
        count = (row["count"] or 0) + 1
        row["count"] = count
        return count <= RATE_LIMIT_MAX_CALLS
    except Exception:
        return True


# ---------------------------------------------------------------------------
# eval_runs logging


def log_request(
    *,
    endpoint: str,
    question: str,
    model: str = "",
    answer: str = "",
    citations: list[dict] | None = None,
    related_links: list[dict] | None = None,
    latency_ms: int = 0,
    cache_stats: dict | None = None,
    api_key_alias: str = "",
    region: str = "",
    error: str = "",
) -> str:
    request_id = uuid.uuid4().hex
    try:
        app_tables.eval_runs.add_row(
            ts=dt.datetime.utcnow(),
            request_id=request_id,
            endpoint=endpoint,
            question=question,
            model=model,
            answer=answer,
            citations=citations or [],
            related_links=related_links or [],
            latency_ms=latency_ms,
            cache_stats=cache_stats or {},
            api_key_alias=api_key_alias,
            region=region,
            error=error,
        )
    except Exception:
        # Best-effort logging — never fail the user request because the
        # log row couldn't be written.
        pass
    return request_id
