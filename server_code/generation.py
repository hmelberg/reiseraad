"""Claude call with prompt caching — reise-API Q&A.

Single entry point:
    answer_question(question, k=8, category=None, kind=None) -> dict

No tool-use loop, no repair loop: the model gets the top-k articles up
front and answers in one shot. Keeps latency low (typical: 2-4 s).
"""

from __future__ import annotations

import json
import re

import anvil.secrets
from anthropic import Anthropic

import prompts
import retrieval

DEFAULT_MODEL = "claude-sonnet-4-6"


# The safety footer for reise is about *outdated* advice, not acute
# medical care — travel vaccines aren't a poison-center scenario. We tell
# the user where to get authoritative current info and remind them the
# API's answer is not a substitute for a real consultation.
SAFETY_FOOTER = (
    "\n\nDette er ikke medisinsk rådgivning. Informasjon om reisevaksiner "
    "og utbrudd kan endres raskt — sjekk alltid den aktuelle FHI-siden "
    "(fhi.no/sm/smittevernrad-ved-reiser/) og rådfør deg med fastlege eller "
    "en vaksinasjonsklinikk før reise."
)


def _client() -> Anthropic:
    api_key = anvil.secrets.get_secret("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


def _cached_prefix_block() -> dict:
    return {
        "type": "text",
        "text": prompts.cached_prefix(),
        "cache_control": {"type": "ephemeral"},
    }


_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        return None


def _recover_partial_json(raw: str) -> dict | None:
    if not raw:
        return None
    candidates = _JSON_OBJ_RE.findall(raw)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _ensure_safety_footer(answer: str) -> str:
    """Guarantee the safety footer appears. Idempotent — won't double-
    append if the FHI URL phrase is already present.
    """
    base = (answer or "").rstrip()
    # Anchor on the FHI URL; it's the most specific token in the footer.
    if "fhi.no/sm/smittevernrad-ved-reiser" in base:
        return base
    return (base + SAFETY_FOOTER).strip()


def _article_to_link(a: dict) -> dict:
    return {
        "id": a["id"],
        "title": a.get("title", ""),
        "url": a.get("url", ""),
        "kind": a.get("kind", ""),
        "category": a.get("category", ""),
        "category_label": a.get("category_label", ""),
        "country": a.get("country"),
        "last_updated": a.get("last_updated"),
    }


def _resolve_citations(citations_raw: list, articles_by_id: dict) -> list[dict]:
    """Turn the model's citation list (ids + notes) into full link records.
    Drops any id the model hallucinated.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for c in citations_raw or []:
        if not isinstance(c, dict):
            continue
        aid = c.get("article_id") or ""
        if not aid or aid in seen:
            continue
        seen.add(aid)
        art = articles_by_id.get(aid)
        if art is None:
            continue
        record = _article_to_link(art)
        note = (c.get("note") or "").strip()
        if note:
            record["note"] = note
        out.append(record)
    return out


def answer_question(
    question: str,
    k: int = 8,
    category: str | None = None,
    kind: str | None = None,
) -> dict:
    """Answer a travel-health question and return:

        {
          "answer": str,
          "citations": list[dict],
          "related_links": list[dict],
          "has_direct_coverage": bool,
          "mentions_outbreak": bool,
          "model": str,
          "cache_stats": dict,
        }

    `related_links` is always the top-k retrieval order; `citations` is
    the subset the model actually drew from. Duplicates removed.
    """
    articles = retrieval.search_articles(
        question, k=k, category=category, kind=kind
    )
    articles_by_id = {a["id"]: a for a in articles}

    related_links = [_article_to_link(a) for a in articles]

    if not articles:
        return {
            "answer": _ensure_safety_footer(
                "Jeg fant ingen artikler på fhi.no/sm/smittevernrad-ved-reiser/ "
                "som dekker spørsmålet ditt direkte. Rådfør deg med en "
                "vaksinasjonsklinikk eller fastlege før reise."
            ),
            "citations": [],
            "related_links": related_links,
            "has_direct_coverage": False,
            "mentions_outbreak": False,
            "model": "",
            "cache_stats": {},
        }

    dynamic = prompts.render_retrieved_articles(articles)
    messages = [
        {
            "role": "user",
            "content": [
                _cached_prefix_block(),
                {
                    "type": "text",
                    "text": (
                        f"# Brukerens spørsmål\n\n{question}\n\n"
                        f"{dynamic}\n\n{prompts.ASK_OUTPUT_CONTRACT}"
                    ),
                },
            ],
        }
    ]

    client = _client()
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=800,
        system=prompts.SYSTEM_PROMPT,
        messages=messages,
    )

    text_out = ""
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            text_out = block.text
            break

    parsed = _parse_json_response(text_out) or _recover_partial_json(text_out)
    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)

    if parsed is None:
        # Fall back to raw text as the answer so the user gets something.
        return {
            "answer": _ensure_safety_footer(text_out or ""),
            "citations": [],
            "related_links": related_links,
            "has_direct_coverage": False,
            "mentions_outbreak": False,
            "model": DEFAULT_MODEL,
            "cache_stats": usage,
        }

    citations = _resolve_citations(parsed.get("citations") or [], articles_by_id)
    has_direct = bool(parsed.get("has_direct_coverage"))
    mentions_outbreak = bool(parsed.get("mentions_outbreak"))

    return {
        "answer": _ensure_safety_footer(parsed.get("answer") or ""),
        "citations": citations,
        "related_links": related_links,
        "has_direct_coverage": has_direct,
        "mentions_outbreak": mentions_outbreak,
        "model": DEFAULT_MODEL,
        "cache_stats": usage,
    }
