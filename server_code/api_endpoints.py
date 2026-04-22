"""HTTP endpoints exposed by the reise-API Anvil app.

All endpoints are reachable at:

    https://<app>.anvil.app/_/api<path>

All endpoints (except /health) require a valid X-API-Key header.
Responses are JSON; 2xx on success, 4xx on client errors (bad auth,
rate limit, bad body), 5xx on unexpected server errors.

Endpoints:
    POST /ask            — fritekst-spørsmål → {answer, citations,
                           related_links, has_direct_coverage, ...}
    GET  /search         — BM25-oppslag uten LLM (debug / søkeboks i UI)
    GET  /health         — liveness probe (ingen auth)
"""

from __future__ import annotations

import json
import time

import anvil.server
from anvil.server import HttpResponse

import generation
import retrieval
import utils


def _json(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(body, ensure_ascii=False),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def _load_body() -> dict:
    """Read the request body as JSON, tolerating non-UTF-8 encodings.

    Windows curl sometimes emits cp1252 for Norwegian characters in `-d`
    payloads, which manifests as an opaque 500 via Anvil's `body_json`.
    We fall back to cp1252 / latin-1 so the handler still responds with
    a useful JSON error.
    """
    req = anvil.server.request
    try:
        body = req.body_json
    except Exception:
        body = None
    if body is None and req.body:
        try:
            raw = req.body.get_bytes()
        except Exception:
            raw = b""
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                body = json.loads(raw.decode(enc))
                break
            except Exception:
                continue
    return body or {}


def _authenticate_or_fail():
    req = anvil.server.request
    alias = utils.authenticate(req)
    if not alias:
        return None, _json({"error": "invalid or missing X-API-Key"}, status=401)
    if not utils.check_rate_limit(alias):
        return None, _json({"error": "rate limit exceeded"}, status=429)
    return alias, None


# Valid category/kind values — mirrors categories.py on the scraper side.
_VALID_CATEGORIES = {
    "afrika", "asia", "europa", "nord-og-mellom-amerika",
    "oseania", "sor-amerika", "tema", "utbrudd",
}
_VALID_KINDS = {"country", "theme", "outbreak"}


def _validate_filters(body_or_kwargs) -> tuple[str | None, str | None, dict | None]:
    """Return (category, kind, error_response). error_response is a
    _json(...) if validation fails; otherwise None.
    """
    cat = (body_or_kwargs.get("category") or "").strip() or None
    knd = (body_or_kwargs.get("kind") or "").strip() or None
    if cat and cat not in _VALID_CATEGORIES:
        return None, None, _json(
            {"error": f"invalid 'category' — must be one of {sorted(_VALID_CATEGORIES)}"},
            status=400,
        )
    if knd and knd not in _VALID_KINDS:
        return None, None, _json(
            {"error": f"invalid 'kind' — must be one of {sorted(_VALID_KINDS)}"},
            status=400,
        )
    return cat, knd, None


# ---------------------------------------------------------------------------
# /ask


@anvil.server.http_endpoint("/ask", methods=["POST"], cross_site_session=False, enable_cors=True)
def http_ask():
    alias, err = _authenticate_or_fail()
    if err:
        return err

    body = _load_body()
    question = (body.get("question") or "").strip()
    if not question:
        return _json({"error": "missing 'question'"}, status=400)
    try:
        k = int(body.get("k", 8))
    except (TypeError, ValueError):
        k = 8
    k = max(3, min(k, 15))

    category, kind, verr = _validate_filters(body)
    if verr:
        return verr

    t0 = time.time()
    try:
        result = generation.answer_question(
            question=question, k=k, category=category, kind=kind
        )
    except Exception as exc:
        latency_ms = int((time.time() - t0) * 1000)
        utils.log_request(
            endpoint="/ask",
            question=question,
            latency_ms=latency_ms,
            api_key_alias=alias,
            region=category or "",
            error=f"{type(exc).__name__}: {exc}",
        )
        return _json({"error": "internal error", "detail": str(exc)}, status=500)
    latency_ms = int((time.time() - t0) * 1000)

    utils.log_request(
        endpoint="/ask",
        question=question,
        model=result.get("model", ""),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        related_links=result.get("related_links", []),
        latency_ms=latency_ms,
        cache_stats=result.get("cache_stats") or {},
        api_key_alias=alias,
        region=category or "",
    )
    result["latency_ms"] = latency_ms
    return _json(result)


# ---------------------------------------------------------------------------
# /search  (BM25 only, no LLM — useful for a search-as-you-type UI)


@anvil.server.http_endpoint("/search", methods=["GET"], cross_site_session=False, enable_cors=True)
def http_search(**kwargs):
    alias, err = _authenticate_or_fail()
    if err:
        return err
    q = (kwargs.get("q") or "").strip()
    if not q:
        return _json({"error": "missing 'q'"}, status=400)
    try:
        k = int(kwargs.get("k", 10))
    except (TypeError, ValueError):
        k = 10
    k = max(1, min(k, 25))

    category, kind, verr = _validate_filters(kwargs)
    if verr:
        return verr

    hits = retrieval.server_search_articles(
        query=q, k=k, category=category, kind=kind
    )
    return _json({"results": hits})


# ---------------------------------------------------------------------------
# /health  (no auth — simple liveness)


@anvil.server.http_endpoint("/health", methods=["GET"], cross_site_session=False, enable_cors=True)
def http_health():
    try:
        cats = retrieval.categories()
        n_articles = sum(c.get("count", 0) for c in cats)
        by_kind: dict[str, int] = {}
        for c in cats:
            by_kind[c.get("kind", "other")] = by_kind.get(c.get("kind", "other"), 0) + c.get("count", 0)
        return _json({
            "status": "ok",
            "articles": n_articles,
            "categories": len(cats),
            "by_kind": by_kind,
        })
    except Exception as exc:
        return _json({"status": "degraded", "error": str(exc)}, status=503)
