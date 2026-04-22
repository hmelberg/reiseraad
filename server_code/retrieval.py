"""BM25 retrieval + synonym expansion over fhi.no travel-advice articles.

Reads its corpus from Anvil Data Files (not Data Tables):

    corpus.pkl     — pickled dict with `articles`, `categories`, and a pre-
                     tokenised BM25 corpus. Produced by
                     seed/build_data_files.py.
    synonyms.json  — hand-authored domain synonyms.

At server-module import the files are loaded once; BM25Okapi is constructed
from the pre-tokenised corpus (fast — no stemming at startup).

Call `reload_data_files()` after uploading a new corpus.pkl to pick up
changes without restarting the worker.

Filters supported:
  - `category`  — region slug (afrika/asia/europa/...) or "tema"/"utbrudd"
  - `kind`      — "country" | "theme" | "outbreak"
  - `country`   — exact country name (case-insensitive)
"""

from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass

import anvil.server
from anvil.files import data_files
from rank_bm25 import BM25Okapi

try:
    import snowballstemmer
    _NO_STEMMER = snowballstemmer.stemmer("norwegian")
    _EN_STEMMER = snowballstemmer.stemmer("english")
except ImportError:  # pragma: no cover
    _NO_STEMMER = None
    _EN_STEMMER = None


_TOKEN_RE = re.compile(r"[a-zA-ZæøåÆØÅ0-9]{2,}", re.UNICODE)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text.lower())


def tokenize(text: str) -> list[str]:
    """Must match seed/build_data_files.py::tokenize exactly."""
    raw = _TOKEN_RE.findall(_fold(text))
    if not raw:
        return []
    if _NO_STEMMER is None:
        return raw
    return _NO_STEMMER.stemWords(raw) + _EN_STEMMER.stemWords(raw)


# ---------------------------------------------------------------------------
# Module-level state


@dataclass
class _Index:
    bm25: BM25Okapi | None
    docs: list[dict]


_corpus: dict | None = None
_articles_index: _Index = _Index(bm25=None, docs=[])
_synonyms_cache: dict[str, list[str]] = {}
_bigram_synonyms_cache: dict[str, list[str]] = {}
_articles_by_id: dict[str, dict] = {}
_country_name_index: dict[str, list[int]] = {}  # folded-name → doc indexes


def _load_corpus() -> None:
    global _corpus, _articles_index, _synonyms_cache, _bigram_synonyms_cache
    global _articles_by_id, _country_name_index

    with open(data_files["corpus.pkl"], "rb") as f:
        _corpus = pickle.load(f)

    bm25_tokens = (_corpus.get("bm25") or {}).get("articles") or []
    _articles_index = _Index(
        bm25=BM25Okapi(bm25_tokens) if bm25_tokens else None,
        docs=_corpus.get("articles") or [],
    )
    _articles_by_id = {a["id"]: a for a in _articles_index.docs}

    # Build country-name index so a query mentioning a country can boost
    # its page(s) regardless of BM25 length normalization.
    _country_name_index = {}
    for idx, a in enumerate(_articles_index.docs):
        name = a.get("country") or ""
        if name:
            _country_name_index.setdefault(_fold(name), []).append(idx)
            # Index the title-from-slug form too (e.g. "sor-afrika" →
            # sør-afrika): we accept a folded match on the URL slug if
            # present in the id.
        title = a.get("title") or ""
        if title and title != name:
            _country_name_index.setdefault(_fold(title), []).append(idx)

    # Synonyms — split into single-token and bigram maps so the query
    # expander can do cheap two-word lookups (e.g. "yellow fever", "south
    # africa") without combinatorial blowup.
    try:
        with open(data_files["synonyms.json"], "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        rows = []
    syn: dict[str, list[str]] = {}
    bigram_syn: dict[str, list[str]] = {}
    for row in rows:
        term = (row.get("term") or "").lower()
        syns = row.get("synonyms") or []
        if not term or not syns:
            continue
        if " " in term:
            bigram_syn.setdefault(term, []).extend(syns)
        else:
            syn.setdefault(term, []).extend(syns)
    _synonyms_cache = syn
    _bigram_synonyms_cache = bigram_syn


def _ensure_loaded() -> None:
    if _corpus is None:
        _load_corpus()


@anvil.server.callable
def reload_data_files() -> dict:
    """Re-read corpus.pkl + synonyms.json after a fresh upload.

    Also busts the cached prompt prefix so any new content (category
    summaries) takes effect on the next request.
    """
    _load_corpus()
    try:
        import prompts
        prompts.refresh_cached_prefix()
    except Exception:
        pass
    return {
        "articles": len(_articles_index.docs),
        "categories": len((_corpus or {}).get("categories") or []),
        "synonyms": len(_synonyms_cache),
        "bigram_synonyms": len(_bigram_synonyms_cache),
        "countries_indexed": len(_country_name_index),
        "schema_version": (_corpus or {}).get("schema_version", 1),
    }


# ---------------------------------------------------------------------------
# Accessors used by prompts/endpoints


def get_corpus() -> dict:
    _ensure_loaded()
    assert _corpus is not None
    return _corpus


def articles_by_id() -> dict[str, dict]:
    _ensure_loaded()
    return _articles_by_id


def categories() -> list[dict]:
    _ensure_loaded()
    return (_corpus or {}).get("categories") or []


# ---------------------------------------------------------------------------
# Query helpers


def expand_synonyms(query: str) -> list[str]:
    _ensure_loaded()
    folded = _fold(query)
    tokens = _TOKEN_RE.findall(folded)
    expanded: list[str] = list(tokens)
    # Two-word lookups first ("yellow fever", "south africa").
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        for syn in _bigram_synonyms_cache.get(bigram, []):
            expanded.append(syn.lower())
    for tok in tokens:
        for syn in _synonyms_cache.get(tok, []):
            expanded.append(syn.lower())
    return expanded


def detect_countries_in_query(query: str) -> list[int]:
    """Return article indexes whose country or title exactly matches a
    contiguous span of the query (folded, case-insensitive).

    This is the main lever for routing a query like "vaksiner til
    thailand" to the Thailand page even when BM25 would otherwise prefer
    a shorter outbreak snippet. We match on country names (preferring
    longer multi-word matches first) and also return exact title matches.
    """
    _ensure_loaded()
    folded = _fold(query)
    hits: list[int] = []
    seen: set[int] = set()
    # Sort known country tokens by length, longest first, so "Nord-Korea"
    # beats "Korea" when both appear.
    keys = sorted(_country_name_index.keys(), key=len, reverse=True)
    for key in keys:
        if not key:
            continue
        # Use word-boundary-ish check — require the key to appear
        # flanked by non-letter chars (or start/end) to avoid matching
        # "iran" inside "iranske".
        pattern = r"(^|[^a-zæøå])" + re.escape(key) + r"($|[^a-zæøå])"
        if re.search(pattern, folded):
            for idx in _country_name_index[key]:
                if idx not in seen:
                    hits.append(idx)
                    seen.add(idx)
    return hits


# ---------------------------------------------------------------------------
# Ranking


def _bm25_top_k(idx: _Index, query_tokens: list[str], k: int) -> list[tuple[float, int]]:
    if not idx.bm25 or not query_tokens:
        return []
    scores = idx.bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    out: list[tuple[float, int]] = []
    for i, s in ranked[:k]:
        if s <= 0:
            break
        out.append((float(s), i))
    return out


# Multiplicative boost applied to a doc's BM25 score when the user's query
# mentions that doc's country by name. Large enough to promote a long
# country page above a short outbreak page with incidental keyword overlap;
# not so large that a genuinely on-topic theme/outbreak page gets buried
# (those still appear in the related_links list, just lower).
_COUNTRY_MATCH_BOOST = 3.0


def search_articles(
    query: str,
    k: int = 8,
    category: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    """Return the top-k most relevant articles for `query`.

    Each row is the full article dict (title, url, sections, body_text,
    category, ...) so callers can both rank and cite.

    If `category` is a region slug / "tema" / "utbrudd", results are
    filtered to that category. If `kind` is set ("country"/"theme"/
    "outbreak"), results are filtered to that kind.

    When the query names a country explicitly, the matching country
    page(s) get a multiplicative boost so they aren't buried under
    shorter, accidentally-matching pages.
    """
    _ensure_loaded()
    tokens = tokenize(" ".join(expand_synonyms(query)))
    # Fetch a wider candidate set so post-filters still have enough to work.
    multiplier = 3 if (category or kind) else 1
    raw_hits = _bm25_top_k(_articles_index, tokens, k=k * multiplier)

    # Country-name boost.
    country_hits = set(detect_countries_in_query(query))
    if country_hits:
        scored: list[tuple[float, int]] = []
        included_country = set()
        for score, idx in raw_hits:
            if idx in country_hits:
                scored.append((score * _COUNTRY_MATCH_BOOST, idx))
                included_country.add(idx)
            else:
                scored.append((score, idx))
        # Make sure the matched country page(s) appear at all, even if
        # BM25 didn't retrieve them. Inject with a floor score based on
        # the max seen.
        if scored:
            floor = max(s for s, _ in scored) * 0.9
        else:
            floor = 1.0
        for idx in country_hits:
            if idx not in included_country:
                scored.append((floor, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        raw_hits = scored

    # Apply filters.
    def _pass(idx: int) -> bool:
        doc = _articles_index.docs[idx]
        if category and doc.get("category") != category:
            return False
        if kind and doc.get("kind") != kind:
            return False
        return True

    hits = [(s, _articles_index.docs[i]) for s, i in raw_hits if _pass(i)]
    return [d for _, d in hits[:k]]


# ---------------------------------------------------------------------------
# Server-callable variants (used by /search endpoint)


@anvil.server.callable
def server_search_articles(
    query: str,
    k: int = 8,
    category: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    hits = search_articles(query=query, k=k, category=category, kind=kind)
    return [
        {
            "id": h["id"],
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "kind": h.get("kind", ""),
            "category": h.get("category", ""),
            "category_label": h.get("category_label", ""),
            "country": h.get("country"),
            "last_updated": h.get("last_updated"),
            "vaccine_hints": h.get("vaccine_hints") or [],
            "disease_hints": h.get("disease_hints") or [],
        }
        for h in hits
    ]
