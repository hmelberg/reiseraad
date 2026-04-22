"""Static prompt content + assembly helpers for the reise-API.

The cached prefix (system prompt + safety rules + kind/category overview)
is built once per worker from the in-memory corpus and re-used across
requests with Anthropic's cache_control. Per-request dynamic content
(retrieved articles, user question) is appended outside the cache
boundary.

Accuracy focus (echoed in SYSTEM_PROMPT):

  1. Answer strictly from the articles provided in the user's turn.
     Never introduce facts not present in them — including country-
     specific vaccine lists, dosing, malaria areas, or outbreak status.
  2. If the articles do not directly answer the user's question, say so
     — don't guess.
  3. Cite the article id(s) you actually relied on.
  4. Always remind the user in the answer that outbreak/vaccine info can
     change and that final decisions should go through a vaccine clinic
     or fastlege.
"""

from __future__ import annotations

import retrieval


# ---------------------------------------------------------------------------
# Static content — stable across requests, lives inside the cache boundary

SYSTEM_PROMPT = """\
Du er en assistent som svarer på spørsmål om smittevern og reisevaksiner
basert KUN på artikler fra Folkehelseinstituttet (fhi.no), som du får
siterte utdrag fra i brukerens tur.

Kjerneregler:

1. Svar KUN basert på de siterte FHI-artiklene i brukerens tur. Du skal
   aldri finne på vaksineanbefalinger, malariaråd, utbruddstatuser eller
   landsspesifikke detaljer som ikke står i de vedlagte utdragene.
2. Hvis artiklene ikke dekker spørsmålet direkte, si det tydelig og
   henvis brukeren videre — til en vaksinasjonsklinikk, fastlege, eller
   fhi.no/sm/smittevernrad-ved-reiser/. Det er aldri galt å si "jeg har
   ikke nok informasjon".
3. Vær presis med tall og begreper. Skriv aldri "ca." eller "omtrent"
   der artikkelen er eksplisitt, og skriv aldri konkrete tall der
   artikkelen ikke har dem.
4. Svar alltid på norsk bokmål, selv om spørsmålet er på engelsk.
5. Hvis spørsmålet gjelder et land eller utbrudd du ikke har fått en
   artikkel om, si at du ikke har kilde for det landet i dette oppslaget
   — ikke ekstrapoler fra nabolandet.
6. Nevn aldri "ring 113" eller andre akuttnumre med mindre artikkelen
   eksplisitt sier det — reisevaksinering er ikke en akuttsituasjon. En
   standardisert disclaimer om at rådene kan endre seg legges til
   automatisk etter ditt svar.
7. Når du siterer, oppgi artikkelens `id`-felt (du ser det i hver
   `<article id="...">`-tag). Ikke finn på id-er.

Format: Svar med et JSON-objekt som matcher kontrakten i brukerens tur.
Ingen prosa utenfor JSON.
"""


SAFETY_RULES = """\
## Sikkerhetsregler (viktig for nøyaktighet)

- **Ingen ekstrapolering.** Hvis artikkelen sier "anbefalt for mange",
  IKKE skriv "anbefales for alle". Hvis artikkelen ramser opp konkrete
  sykdommer, IKKE legg til andre som kan virke relevante.
- **Ingen selvstendige medisinske vurderinger.** Du verifiserer ikke om
  en bestemt vaksine er trygg for brukeren selv — si at brukeren bør
  avklare det med lege eller vaksinasjonsklinikk.
- **Utbrudd er tidskritiske.** Hvis svaret berører et utbrudd eller en
  nylig endret anbefaling, flagg eksplisitt at situasjonen kan ha
  endret seg siden artikkelen ble oppdatert (se `last_updated` i
  artikkelens metadata).
- **Usikkerhet.** Hvis artiklene gir motstridende eller utdatert
  informasjon, si det. Bedre å si "informasjonen jeg har er fra
  <dato>" enn å late som du har en definitiv fasit.
- **Sertifikatkrav.** Hvis brukeren spør om gulfebersertifikat og
  artikkelen sier det er krav, si det presist; hvis ikke nevnt, si at
  dette ikke fremgår av artikkelen og at brukeren bør sjekke med
  ambassaden før reise.
- **Barn, gravide, immunsupprimerte:** Gi KUN råd som er eksplisitt i
  artikkelen. Disse gruppene har særregler — ikke generaliser.
"""


RESPONSE_STYLE = """\
## Stil på svaret

- 3-6 korte setninger. Konkret, nøkternt, på norsk bokmål.
- Nevn landet og relevante vaksiner/sykdommer ved navn tidlig i svaret.
- Hvis det finnes utbruddsvarsler i de siterte artiklene, nevn det
  kort som en egen setning.
- Ingen punktlister eller markdown i selve `answer`-feltet.
- Bruk aktivt språk ("Anbefalingen er MMR og hepatitt A" — ikke "det
  anbefales vanligvis at").
- Gjenta ikke brukerens spørsmål. Gå rett på svaret.
- Hvis du ikke har dekning i artiklene: si det i én tydelig setning.
"""


def build_kinds_overview() -> str:
    """Render the corpus layout so the model knows what it can cite."""
    cats = retrieval.categories()
    if not cats:
        return ""
    by_kind: dict[str, list[dict]] = {}
    for c in cats:
        by_kind.setdefault(c.get("kind", "other"), []).append(c)

    lines = ["## Kildesiden (FHI)", "",
             "Artiklene er hentet fra fhi.no/sm/smittevernrad-ved-reiser/ "
             "og dekker disse typene innhold (antall i parentes):", ""]
    if "country" in by_kind:
        totals = sum(c["count"] for c in by_kind["country"])
        lines.append(f"- **Landsider** ({totals} land) organisert i regioner:")
        for c in by_kind["country"]:
            lines.append(f"    - {c.get('label','')} (`{c.get('slug','')}`, {c.get('count',0)} land)")
    if "theme" in by_kind:
        totals = sum(c["count"] for c in by_kind["theme"])
        lines.append(f"- **Temasider** ({totals} artikler): generelle råd "
                     "(gravide, rabies, mmr, god reise m.m.)")
    if "outbreak" in by_kind:
        totals = sum(c["count"] for c in by_kind["outbreak"])
        lines.append(f"- **Utbruddsmeldinger** ({totals} artikler): "
                     "tidskritiske varsler om utbrudd i utlandet.")
    lines.append("")
    lines.append("Brukeren ser aldri noe som ikke står i artiklene — "
                 "du skal ikke bruke allmennkunnskap om vaksiner eller "
                 "sykdommer som ikke finnes i de siterte utdragene.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached prefix assembly


_cached_prefix: str | None = None


def cached_prefix() -> str:
    global _cached_prefix
    if _cached_prefix is None:
        _cached_prefix = "\n\n".join(
            filter(
                None,
                [
                    SAFETY_RULES,
                    RESPONSE_STYLE,
                    build_kinds_overview(),
                ],
            )
        )
    return _cached_prefix


def refresh_cached_prefix() -> None:
    """Call after retrieval.reload_data_files() so a new corpus takes effect."""
    global _cached_prefix
    _cached_prefix = None


# ---------------------------------------------------------------------------
# Per-request dynamic content


# Preferred order for section rendering — roughly the order the user
# expects on a country page. Keys come from scraper/parse_article.py's
# section-keyword mapping.
_SECTION_ORDER = [
    "vaksiner",
    "malaria",
    "mygg",
    "flatt",
    "mat_og_vann",
    "diare",
    "bading",
    "sol_og_varme",
    "hoyde",
    "medisin",
    "reiseapotek",
    "blodpropp",
    "dyr",
    "sex",
    "transport",
    "beredskap",
    "tips",
    "annet",
    "beskrivelse",
    "rad",
    "forebygging",
    "symptomer",
    "smitte",
    "utbrudd",
    "gravid",
    "barn",
    "malgruppe",
    "dosering",
    "bivirkning",
]


def render_retrieved_articles(rows: list[dict], max_chars_per_section: int = 1500) -> str:
    """Compose the article payload for the model's context window.

    Each article gets an `<article id="...">` tag so the model can cite by
    id directly. We favour the structured `sections` dict and fall back to
    `body_text` when no sections were parsed (outbreak and some theme pages).
    """
    if not rows:
        return (
            "## Kildeartikler\n\n(Ingen artikler matchet spørsmålet. "
            "Si at du ikke har kilde for dette i ditt svar, og henvis "
            "brukeren videre.)"
        )

    parts = ["## Kildeartikler", ""]
    for r in rows:
        parts.append(f'<article id="{r["id"]}">')
        parts.append(f"**Tittel:** {r.get('title','')}")
        parts.append(f"**Type:** {r.get('kind','')}")
        parts.append(f"**Region/tema:** {r.get('category_label','')}")
        if r.get("country"):
            parts.append(f"**Land:** {r['country']}")
        if r.get("url"):
            parts.append(f"**URL:** {r['url']}")
        if r.get("last_updated"):
            parts.append(f"**Sist oppdatert (FHI):** {r['last_updated']}")
        sections = r.get("sections") or {}
        body_parts: list[str] = []
        if sections:
            # Render in the preferred order, then any leftovers.
            seen: set[str] = set()
            for key in _SECTION_ORDER:
                val = sections.get(key)
                if not val:
                    continue
                trimmed = val if len(val) <= max_chars_per_section else val[:max_chars_per_section] + " …"
                body_parts.append(f"**{key}:** {trimmed}")
                seen.add(key)
            for key, val in sections.items():
                if key in seen or not val:
                    continue
                trimmed = val if len(val) <= max_chars_per_section else val[:max_chars_per_section] + " …"
                body_parts.append(f"**{key}:** {trimmed}")
        if not body_parts:
            fallback = r.get("body_text") or ""
            if len(fallback) > max_chars_per_section:
                fallback = fallback[:max_chars_per_section] + " …"
            body_parts.append(fallback)
        parts.append("\n\n".join(body_parts))
        parts.append("</article>")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output contract

ASK_OUTPUT_CONTRACT = """\
Svar med et JSON-objekt på én linje (ingen markdown-fencing, ingen ekstra
prosa utenfor JSON):

{
  "answer": "3-6 korte setninger med svaret, på norsk bokmål",
  "citations": [
    {"article_id": "<id fra en <article id='...'> tag>", "note": "kort hint, f.eks. 'anbefalte vaksiner' eller 'malaria'"}
  ],
  "has_direct_coverage": true | false,
  "mentions_outbreak": true | false
}

Feltforklaring:
- `answer`: se stilreglene over. Maks 6 setninger.
- `citations`: KUN artikkel-id-er du faktisk brukte — maks 4. Bruk eksakt
  streng fra `<article id="...">`. Ikke finn på id-er.
- `has_direct_coverage`: true hvis artiklene gir et tydelig, dekkende
  svar på spørsmålet. false hvis du bare kunne gi et delvis svar og
  nevnte at informasjonen er begrenset.
- `mentions_outbreak`: true hvis svaret ditt refererer til et pågående
  utbrudd (da skal klienten fremheve utbruddsfeltet).
"""
