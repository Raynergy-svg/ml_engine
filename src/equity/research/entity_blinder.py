"""Entity-blinder — the PRIMARY lookahead control for Track B research-alpha.

See ``docs/experiment-equity-research-alpha-prereg-2026-06-30.md`` §1.1 and the
frozen :mod:`src.equity.research.contracts` spine.

The threat model is adversarial: an LLM asked to "research" a filing already knows
that issuer's future from pretraining. This module strips identifying signal so the
research scorer cannot map the text back to a remembered outcome. We assume a *smart*
reader will try to re-identify the issuer, so the design is:

* **Deny-list first (precise).** The caller supplies what EDGAR knows about the
  issuer — company name(s) incl. former names, ticker(s), CIK, exchange, executive
  / board names, brand / product names, HQ city / state. Each distinct entity maps
  to a STABLE neutral placeholder (``COMPANY_A``, ``TICKER_A``, ``PERSON_1``,
  ``PLACE_1``, ``PRODUCT_1``) by first-appearance order, so the blinded text stays
  coherent within one document (every "Apple" becomes the same ``COMPANY_A``).
* **Future-date scrub.** Any date on or after ``as_of`` becomes ``[REDACTED_DATE]``
  (a filing must never reveal future-dated text). Dates strictly before ``as_of``
  are KEPT — they are part of the legitimate point-in-time record.
* **Residual-risk heuristic (honest, imperfect).** A capitalized proper-noun scan
  reports Title-Case multiword tokens that survived the deny-list — these are the
  re-identification leaks a verifier samples for. Deny-list + heuristic is NOT a
  guarantee; the audit surfaces what slipped through so the verifier can see it.

Pure offline module: no network, no LLM, no RNG. Deterministic — same input maps to
byte-identical output (placeholder numbering is stable by first-appearance order).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from src.equity.research.contracts import BlindedText, FilingText

# ---------------------------------------------------------------------------
# Placeholder prefixes per redaction category. The suffix is assigned by
# first-appearance order: COMPANY_A, COMPANY_B, ... / PERSON_1, PERSON_2, ...
# Companies/tickers/places use letters; people/products use numbers — purely a
# readability convention so a blinded reader can still follow "COMPANY_A vs
# COMPANY_B" while never recovering the real names.
# ---------------------------------------------------------------------------
CAT_COMPANY = "company"
CAT_TICKER = "ticker"
CAT_CIK = "cik"
CAT_EXCHANGE = "exchange"
CAT_PERSON = "person"
CAT_PRODUCT = "product"
CAT_PLACE = "place"
CAT_DATE = "date"

# (placeholder prefix, numbering style) per category.
_PLACEHOLDER_SPEC: Dict[str, Tuple[str, str]] = {
    CAT_COMPANY: ("COMPANY_", "alpha"),
    CAT_TICKER: ("TICKER_", "alpha"),
    CAT_CIK: ("CIK_", "alpha"),
    CAT_EXCHANGE: ("EXCHANGE_", "alpha"),
    CAT_PERSON: ("PERSON_", "numeric"),
    CAT_PRODUCT: ("PRODUCT_", "numeric"),
    CAT_PLACE: ("PLACE_", "numeric"),
}
_DATE_PLACEHOLDER = "[REDACTED_DATE]"

# Categories that scrub raw issuer-supplied strings via deny-list matching.
# Order matters ONLY for readability of the audit, not correctness (each entity
# gets its own placeholder regardless of category processing order).
_DENYLIST_CATEGORIES: Tuple[str, ...] = (
    CAT_COMPANY,
    CAT_PERSON,
    CAT_PRODUCT,
    CAT_PLACE,
    CAT_EXCHANGE,
    CAT_TICKER,
    CAT_CIK,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# "March 3, 2025" / "Mar. 3 2025" / "3 March 2025".
_RE_DATE_MDY = re.compile(
    r"\b(?P<month>" + _MONTH_ALT + r")\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_RE_DATE_DMY = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<month>" + _MONTH_ALT + r")\.?,?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
# ISO 2025-03-03 and slashed 3/3/2025 or 03-03-2025.
_RE_DATE_ISO = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_RE_DATE_SLASH = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{4})\b"
)
# "March 2025" (month + year, no day).
_RE_DATE_MY = re.compile(
    r"\b(?P<month>" + _MONTH_ALT + r")\.?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
# Bare four-digit year, treated as a date only when contextually so (handled in
# code by the year-threshold check, not blindly).
_RE_BARE_YEAR = re.compile(r"\b(?P<year>\d{4})\b")

# Title-Case multiword proper-noun sequences for the residual-risk scan. Matches
# runs of >=1 capitalized word (optionally joined by &, of, and, the, for) —
# e.g. "Foobar Dynamics", "Bank of America". Single capitalized words at the
# start of sentences are noisy, so we only flag sequences of >=2 capitalized
# tokens OR a single ALL-CAPS token of length >=3 (possible ticker-like leak).
_RE_PROPER_SEQ = re.compile(
    r"\b[A-Z][a-zA-Z0-9.&'-]*"
    r"(?:\s+(?:of|and|the|for|&)\s+[A-Z][a-zA-Z0-9.&'-]*"
    r"|\s+[A-Z][a-zA-Z0-9.&'-]*)+\b"
)
_RE_ALLCAPS = re.compile(r"\b[A-Z]{3,6}\b")

# Common words that legitimately appear Title-Cased (sentence starts, headings)
# and should NOT be treated as re-identification leaks. Lowercased for matching.
_RESIDUAL_STOPWORDS = frozenset({
    "the", "this", "that", "these", "those", "we", "our", "company", "form",
    "item", "note", "notes", "part", "during", "as", "of", "in", "on", "for",
    "to", "and", "or", "report", "annual", "quarterly", "fiscal", "year",
    "quarter", "results", "operations", "financial", "statements", "business",
    "risk", "factors", "management", "discussion", "analysis", "table",
    "contents", "exhibit", "signature", "signatures", "united", "states",
    "securities", "exchange", "commission", "washington", "march", "june",
    "september", "december", "january", "february", "april", "may", "july",
    "august", "october", "november",
})

# Numeric FINGERPRINTS — comma-grouped figures (>=4 digits) and $-magnitudes are
# near-unique per issuer (e.g. "$394,328 million" -> Apple FY22). A pretrained
# model re-identifies from these even with every proper noun redacted, and the
# deny-list never touches numbers, so the residual scan MUST surface them or its
# count=0 gives false comfort (2026-06-30 review CRITICAL-1).
_RE_NUMERIC_FINGERPRINT = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?"          # $394,328
    r"|\$\s?\d+(?:\.\d+)?\s*(?:million|billion|thousand)\b"  # $1.5 billion
    r"|\b\d{1,3}(?:,\d{3}){1,}(?:\.\d+)?\b",      # 394,328
    re.IGNORECASE,
)

# US state / common state-of-incorporation names. A bare surviving state (e.g.
# "incorporated in California") is a single-token leak the Title-Case multiword
# scan misses; this fixed list keeps it LOW-noise (50 states) but caught.
_US_STATES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "ohio", "oklahoma", "oregon",
    "pennsylvania", "tennessee", "texas", "utah", "vermont", "virginia",
    "wisconsin", "wyoming",
})
_RE_STATE = re.compile(
    r"\b(" + "|".join(sorted(_US_STATES)) + r")\b", re.IGNORECASE
)


def _dedup_surface_forms(terms: Sequence[str]) -> List[str]:
    """Strip / drop blanks / case-insensitively dedup, then sort longest-first.

    Longest-first (ties broken alphabetically) is required so a specific surface
    form is redacted before a substring of it, and is also deterministic.
    """
    cleaned: List[str] = []
    seen: set = set()
    for t in terms:
        s = (t or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    cleaned.sort(key=lambda x: (-len(x), x.lower()))
    return cleaned


def _group_aliases(surface_forms: List[str]) -> List[Tuple[str, str]]:
    """Collapse alias surface forms of one entity to a shared canonical key.

    Input is already longest-first. A shorter form that is a word-bounded,
    case-insensitive substring of a longer form already assigned to an entity
    joins that entity (so "Cupertino" folds into "Cupertino, California"); a form
    matching no existing entity starts a new one keyed by its own longest surface
    form. Returns ``(canonical_key, surface_form)`` pairs, still longest-first so
    redaction ordering is preserved.
    """
    canonical_of: Dict[str, str] = {}
    for s in surface_forms:  # longest-first
        low = s.lower()
        assigned: Optional[str] = None
        for longer, canon in canonical_of.items():
            # `longer` was seen earlier so it is >= length of `s`.
            if _is_word_substring(low, longer):
                assigned = canon
                break
        canonical_of[low] = assigned if assigned is not None else low
    return [(canonical_of[s.lower()], s) for s in surface_forms]


def _is_word_substring(needle_low: str, haystack_low: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` on word boundaries (both lower)."""
    if needle_low == haystack_low:
        return True
    pattern = r"(?<![\w])" + re.escape(needle_low) + r"(?![\w])"
    return re.search(pattern, haystack_low) is not None


@dataclass(frozen=True)
class IssuerBundle:
    """Issuer metadata the caller supplies from EDGAR for deny-list redaction.

    Every field is OPTIONAL — the caller redacts what it knows. Empty / blank
    strings are ignored. ``former_company_names`` covers historical names that an
    old filing may use (and that pretraining may still map to the issuer).
    """

    company_names: Sequence[str] = field(default_factory=tuple)
    former_company_names: Sequence[str] = field(default_factory=tuple)
    tickers: Sequence[str] = field(default_factory=tuple)
    cik: Optional[str] = None
    exchanges: Sequence[str] = field(default_factory=tuple)
    person_names: Sequence[str] = field(default_factory=tuple)
    product_names: Sequence[str] = field(default_factory=tuple)
    hq_locations: Sequence[str] = field(default_factory=tuple)

    def terms_by_category(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return ``(canonical_key, surface_form)`` deny-list pairs per category.

        A bundle describes ONE issuer, so surface forms that name the same entity
        must collapse to the SAME placeholder. Two grouping disciplines:

        * **Single-entity categories** (company / ticker / cik): every supplied
          surface form is an alias of the one issuer, so they ALL share one
          canonical key — "Apple Inc.", "Apple, Inc." and "Apple" all map to
          ``COMPANY_A``; "AAPL" -> ``TICKER_A``; every CIK form -> ``CIK_A``.
        * **Multi-entity categories** (person / product / place / exchange):
          genuinely distinct entities (two executives, two products) must get
          distinct placeholders, but alias surface forms of ONE entity
          ("Cupertino" vs "Cupertino, California") must still collapse. We group
          by case-insensitive substring containment: a shorter form that is a
          word-bounded substring of a longer already-seen form joins that
          entity; otherwise it starts a new one.

        Within each category the pairs are ordered LONGEST-surface-FIRST so the
        redaction pass rewrites "Apple, Inc." before its substring "Apple" — else
        the substring pass would leave an orphan "COMPANY_A, Inc." re-leaking the
        legal suffix.
        """
        ciks: List[str] = []
        if self.cik is not None and str(self.cik).strip():
            raw = str(self.cik).strip()
            ciks.append(raw)
            # Also redact the zero-padded 10-digit EDGAR form and the
            # un-padded integer form, since filings use both.
            digits = re.sub(r"\D", "", raw)
            if digits:
                ciks.append(digits)
                ciks.append(digits.zfill(10))
                ciks.append(str(int(digits)))

        single_entity = {
            CAT_COMPANY: list(self.company_names) + list(self.former_company_names),
            CAT_TICKER: list(self.tickers),
            CAT_CIK: ciks,
        }
        multi_entity = {
            CAT_EXCHANGE: list(self.exchanges),
            CAT_PERSON: list(self.person_names),
            CAT_PRODUCT: list(self.product_names),
            CAT_PLACE: list(self.hq_locations),
        }

        out: Dict[str, List[Tuple[str, str]]] = {}
        for cat, terms in single_entity.items():
            cleaned = _dedup_surface_forms(terms)
            # One canonical key for the whole single-issuer category.
            out[cat] = [(cat, s) for s in cleaned]
        for cat, terms in multi_entity.items():
            out[cat] = _group_aliases(_dedup_surface_forms(terms))
        return out


class _PlaceholderRegistry:
    """Assigns stable placeholders by first-appearance order, per category."""

    def __init__(self) -> None:
        # canonical-key -> placeholder, per category.
        self._maps: Dict[str, Dict[str, str]] = {
            cat: {} for cat in _PLACEHOLDER_SPEC
        }
        self._counts: Dict[str, int] = {cat: 0 for cat in _PLACEHOLDER_SPEC}

    @staticmethod
    def _alpha_suffix(n: int) -> str:
        """0 -> A, 1 -> B, ... 25 -> Z, 26 -> AA (Excel-style, deterministic)."""
        out = ""
        n += 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            out = chr(ord("A") + rem) + out
        return out

    def placeholder_for(self, category: str, canonical_key: str) -> str:
        """Return the stable placeholder for a distinct entity in a category."""
        prefix, style = _PLACEHOLDER_SPEC[category]
        cat_map = self._maps[category]
        if canonical_key in cat_map:
            return cat_map[canonical_key]
        idx = self._counts[category]
        if style == "alpha":
            suffix = self._alpha_suffix(idx)
        else:
            suffix = str(idx + 1)
        placeholder = f"{prefix}{suffix}"
        cat_map[canonical_key] = placeholder
        self._counts[category] += 1
        return placeholder

    def all_placeholders(self) -> set:
        out: set = set()
        for cat_map in self._maps.values():
            out.update(cat_map.values())
        out.add(_DATE_PLACEHOLDER.strip("[]"))
        return out


def _parse_as_of(as_of: str) -> date:
    """Parse the ISO ``as_of`` string into a date (strict; surfaces bad input)."""
    try:
        return date.fromisoformat(as_of[:10])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"entity_blinder: as_of={as_of!r} is not an ISO date (YYYY-MM-DD)"
        ) from exc


def _build_denylist_pattern(term: str, *, letter_boundary: bool = False) -> str:
    """Build a regex fragment for one deny-list term.

    Handles: case-insensitivity (caller applies the flag), interior whitespace
    runs (one or more spaces / newlines between words), trailing possessive
    ``'s``, and optional trailing legal-suffix punctuation already embedded in
    the term. Word boundaries guard against substring hits inside other words
    (so "Apple" does not redact "Applesauce").

    ``letter_boundary`` uses LETTER-only boundaries instead of ``\\w`` ones. It
    exists for TICKERS, which routinely glue onto digits/underscores in filenames
    and accession refs ("AAPL_10K", "AAPL10K"): a ``\\w`` boundary treats ``_``
    and digits as word chars and so FAILS to redact the ticker there (a confirmed
    leak, 2026-06-30 review). Letter boundaries break on ``_``/digits, redacting
    the ticker; the only cost is occasional safe over-redaction of a short ticker
    glued to a number ("CAT5"), which is acceptable for a lookahead control.
    """
    # Split on whitespace; rejoin allowing flexible interior whitespace.
    parts = [re.escape(p) for p in term.split()]
    body = r"\s+".join(parts)
    # Left boundary: not a word char before. Right boundary: optional possessive,
    # then a non-word char or end. We use lookarounds so adjacent punctuation
    # (",", ".", "'s") is handled without consuming it destructively.
    left = r"(?<![A-Za-z])" if letter_boundary else r"(?<![\w])"
    right = r"(?![A-Za-z])" if letter_boundary else r"(?![\w])"
    return left + body + r"(?:['’]s)?" + right


def _redact_denylist(
    text: str,
    terms_by_cat: Dict[str, List[Tuple[str, str]]],
    registry: _PlaceholderRegistry,
    counts: Dict[str, int],
) -> str:
    """Replace every deny-list surface form with its entity's stable placeholder.

    Each item is a ``(canonical_key, surface_form)`` pair; surface forms sharing
    a canonical key resolve to the SAME placeholder so the blinded text stays
    coherent (all aliases of one issuer -> one ``COMPANY_A``).
    """
    for category in _DENYLIST_CATEGORIES:
        for canonical_key, term in terms_by_cat.get(category, []):
            pattern = _build_denylist_pattern(
                term, letter_boundary=(category == CAT_TICKER)
            )
            placeholder = registry.placeholder_for(category, canonical_key)

            def _sub(_match: "re.Match[str]", _ph: str = placeholder) -> str:
                return _ph

            new_text, n = re.subn(pattern, _sub, text, flags=re.IGNORECASE)
            if n:
                counts[category] = counts.get(category, 0) + n
                text = new_text
    return text


def _redact_future_dates(
    text: str, as_of: date, counts: Dict[str, int]
) -> str:
    """Replace any date >= ``as_of`` with ``[REDACTED_DATE]``; keep earlier dates.

    Processed most-specific-first (full day-month-year forms, then month-year,
    then bare year). Two protections keep later passes from corrupting earlier
    matches:

    * A date that is REDACTED becomes ``[REDACTED_DATE]`` (digit-free) so later
      regexes cannot re-touch it.
    * A full date that is KEPT (it is strictly before ``as_of`` and therefore
      legitimate PIT history) is wrapped in a digit-free sentinel so the later
      bare-year / month-year passes cannot see its year and wrongly redact it
      (e.g. "September 25, 2021" kept under as_of 2021-10-29 must not lose its
      "2021"). Sentinels are unwrapped at the end, restoring the original text.
    """
    sentinels: Dict[str, str] = {}
    text = _apply_date_regex(
        text, _RE_DATE_MDY, as_of, counts, _mdy_to_date, sentinels)
    text = _apply_date_regex(
        text, _RE_DATE_DMY, as_of, counts, _dmy_to_date, sentinels)
    text = _apply_date_regex(
        text, _RE_DATE_ISO, as_of, counts, _iso_to_date, sentinels)
    text = _apply_date_regex(
        text, _RE_DATE_SLASH, as_of, counts, _slash_to_date, sentinels)
    text = _apply_date_regex(
        text, _RE_DATE_MY, as_of, counts, _my_to_date, sentinels)
    text = _apply_date_regex(
        text, _RE_BARE_YEAR, as_of, counts, _bare_year_to_date, sentinels)
    # Restore kept-date sentinels to their original surface text.
    for token, original in sentinels.items():
        text = text.replace(token, original)
    return text


def _mdy_to_date(m: "re.Match[str]") -> Optional[date]:
    try:
        month = _MONTHS[m.group("month").lower()]
        return date(int(m.group("year")), month, int(m.group("day")))
    except (KeyError, ValueError):
        return None


def _dmy_to_date(m: "re.Match[str]") -> Optional[date]:
    try:
        month = _MONTHS[m.group("month").lower()]
        return date(int(m.group("year")), month, int(m.group("day")))
    except (KeyError, ValueError):
        return None


def _iso_to_date(m: "re.Match[str]") -> Optional[date]:
    try:
        return date(int(m.group("year")), int(m.group("month")),
                    int(m.group("day")))
    except ValueError:
        return None


def _slash_to_date(m: "re.Match[str]") -> Optional[date]:
    # US convention M/D/Y (filings are US-SEC).
    try:
        return date(int(m.group("year")), int(m.group("month")),
                    int(m.group("day")))
    except ValueError:
        return None


def _my_to_date(m: "re.Match[str]") -> Optional[date]:
    # Month + year, no day. Treat as the first of the month for comparison —
    # any month strictly before as_of's month is kept; the as_of month or later
    # is redacted (conservative: a partial future date is still future signal).
    try:
        month = _MONTHS[m.group("month").lower()]
        return date(int(m.group("year")), month, 1)
    except (KeyError, ValueError):
        return None


def _bare_year_to_date(m: "re.Match[str]") -> Optional[date]:
    # A bare year is contextually a date only when it is plausibly a calendar
    # year (1900-2100). January 1 is used for the >= comparison so the WHOLE
    # of as_of's year and later is treated as future-or-current (redacted),
    # while strictly-earlier years are kept.
    try:
        y = int(m.group("year"))
    except ValueError:
        return None
    if y < 1900 or y > 2100:
        return None
    return date(y, 1, 1)


def _apply_date_regex(
    text: str,
    regex: "re.Pattern[str]",
    as_of: date,
    counts: Dict[str, int],
    to_date,
    sentinels: Dict[str, str],
) -> str:
    """Run one date regex over ``text``: redact matches >= as_of, protect kept.

    Bare-year comparison uses Jan-1 of the year vs Jan-1 of as_of's year so the
    entire as_of year is redacted (a 10-K filed in 2025 must not surface "2025"
    or later). Month-year uses the first of the month. Full dates compare exactly
    against ``as_of``.

    A KEPT match that carries an explicit month (everything except the bare-year
    pass) is replaced with a digit-free sentinel and recorded in ``sentinels`` so
    a later, coarser pass cannot re-match its year. Kept bare years are left as
    plain text (nothing coarser follows).
    """
    result: List[str] = []
    last = 0
    is_bare_year = regex is _RE_BARE_YEAR
    is_month_year = regex is _RE_DATE_MY
    for m in regex.finditer(text):
        d = to_date(m)
        if d is None:
            continue
        if is_bare_year:
            redact = d >= date(as_of.year, 1, 1)
        elif is_month_year:
            redact = d >= date(as_of.year, as_of.month, 1)
        else:
            redact = d >= as_of
        result.append(text[last:m.start()])
        if redact:
            result.append(_DATE_PLACEHOLDER)
            counts[CAT_DATE] = counts.get(CAT_DATE, 0) + 1
        elif is_bare_year:
            result.append(m.group(0))  # nothing coarser follows; leave as-is
        else:
            token = _sentinel_token(len(sentinels))
            sentinels[token] = m.group(0)
            result.append(token)
        last = m.end()
    result.append(text[last:])
    return "".join(result)


def _sentinel_token(idx: int) -> str:
    """A digit-free, regex-inert placeholder for a KEPT date pending restore."""
    return f"\x00KEPTDATE{idx}\x00"


def _residual_risk_scan(
    text: str, known_placeholders: set, sample_limit: int = 25
) -> Dict[str, object]:
    """Heuristically flag capitalized proper-noun sequences that SURVIVED.

    This is HOW a verifier samples for re-identification leakage. It is honest
    about being imperfect: deny-list + this heuristic cannot catch every entity
    (a single-word brand the caller never supplied will slip through). We report
    a deduped, deterministically-ordered sample plus a count so the verifier can
    eyeball what leaked.

    A "leak candidate" is a multiword Title-Case run, or a 3-6 char ALL-CAPS
    token (ticker-shaped), that is NOT one of our own placeholders and not a
    boilerplate filing stopword. Placeholders are skipped because COMPANY_A /
    PERSON_1 are intentional, not leaks.
    """
    candidates: List[str] = []
    seen: set = set()

    def _consider(token: str) -> None:
        token = token.strip()
        if not token:
            return
        if token in known_placeholders:
            return
        # Skip if every alpha word is a stopword (pure boilerplate heading).
        words = [w for w in re.split(r"\s+", token) if w]
        if words and all(
            w.lower().strip(".,'&-") in _RESIDUAL_STOPWORDS for w in words
        ):
            return
        # Skip if the token is itself (or contains only) our placeholders.
        if all(
            (w in known_placeholders or w.strip(".,") in known_placeholders)
            for w in words
        ):
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(token)

    for m in _RE_PROPER_SEQ.finditer(text):
        _consider(m.group(0))
    for m in _RE_ALLCAPS.finditer(text):
        tok = m.group(0)
        if tok in known_placeholders:
            continue
        if tok.lower() in _RESIDUAL_STOPWORDS:
            continue
        _consider(tok)

    candidates.sort()

    # Low-noise, high-value single-token leaks the multiword scan misses.
    numeric = sorted({m.group(0).strip() for m in _RE_NUMERIC_FINGERPRINT.finditer(text)})
    states = sorted({m.group(1) for m in _RE_STATE.finditer(text)})

    # The headline: ANY of the three channels firing means "not demonstrably
    # clean". A consumer must read THIS, never leak_candidate_count alone — a
    # count=0 proper-noun scan said nothing about numeric/state fingerprints.
    any_leak_signal = bool(candidates or numeric or states)

    return {
        "leak_candidate_count": len(candidates),
        "leak_candidate_sample": candidates[:sample_limit],
        "numeric_fingerprint_count": len(numeric),
        "numeric_fingerprint_sample": numeric[:sample_limit],
        "survived_state_count": len(states),
        "survived_state_sample": states[:sample_limit],
        "any_leak_signal": any_leak_signal,
        "heuristic": (
            "Three channels: (1) Title-Case multiword + 3-6 char ALL-CAPS "
            "tokens, (2) comma-grouped/$-magnitude numeric fingerprints, "
            "(3) surviving US state names. Deny-list + heuristic is NOT "
            "exhaustive; a single-word brand never supplied by the caller will "
            "not be caught. Read any_leak_signal, NOT leak_candidate_count "
            "alone; treat a True as POSSIBLE (not certain) un-redacted issuer "
            "signal — and note the load-bearing lookahead control is the "
            "post-cutoff arm, not this blinder."
        ),
    }


def blind_filing(filing: FilingText, bundle: IssuerBundle) -> BlindedText:
    """Blind one PIT filing against an issuer deny-list + future-date scrub.

    Args:
        filing: the raw, pre-blinding :class:`FilingText` (carries ``as_of``,
            ``form``, ``filed``, ``text``, plus ``ticker`` / ``cik`` which are
            DROPPED from the output by contract).
        bundle: issuer metadata the caller knows from EDGAR (company / person /
            product / place names, tickers, CIK, exchanges). The bundle's
            ``tickers`` / ``cik`` SHOULD already include ``filing.ticker`` /
            ``filing.cik``; we add them defensively so the filing's own
            identifiers can never survive even if the caller forgot them.

    Returns:
        A :class:`BlindedText` (no ticker / cik) with blinded ``text`` and an
        ``audit`` dict carrying per-category counts, the total, and the
        residual-risk scan.
    """
    as_of_date = _parse_as_of(filing.as_of)

    # Defensive: always include the filing's own ticker/cik in the deny-list,
    # even if the caller's bundle omitted them — they are the single most
    # dangerous re-identifiers and must never survive.
    eff_tickers = list(bundle.tickers)
    if filing.ticker and filing.ticker.strip():
        eff_tickers.append(filing.ticker)
    eff_cik = bundle.cik
    if (eff_cik is None or not str(eff_cik).strip()) and filing.cik:
        eff_cik = filing.cik

    eff_bundle = IssuerBundle(
        company_names=bundle.company_names,
        former_company_names=bundle.former_company_names,
        tickers=eff_tickers,
        cik=eff_cik,
        exchanges=bundle.exchanges,
        person_names=bundle.person_names,
        product_names=bundle.product_names,
        hq_locations=bundle.hq_locations,
    )
    terms_by_cat = eff_bundle.terms_by_category()

    # If the filing CIK differs from a (set) bundle CIK, fold the filing CIK's
    # surface forms into the single-issuer CIK group so BOTH are redacted to the
    # same CIK placeholder (rare, but a former CIK can appear in an old filing).
    if filing.cik and str(filing.cik).strip():
        extra_pairs = IssuerBundle(cik=filing.cik).terms_by_category()[CAT_CIK]
        existing_low = {s.lower() for _, s in terms_by_cat.get(CAT_CIK, [])}
        merged = list(terms_by_cat.get(CAT_CIK, []))
        for _, surface in extra_pairs:
            if surface.lower() not in existing_low:
                merged.append((CAT_CIK, surface))
        merged.sort(key=lambda pair: (-len(pair[1]), pair[1].lower()))
        terms_by_cat[CAT_CIK] = merged

    registry = _PlaceholderRegistry()
    counts: Dict[str, int] = {}

    text = filing.text
    text = _redact_denylist(text, terms_by_cat, registry, counts)
    text = _redact_future_dates(text, as_of_date, counts)

    residual = _residual_risk_scan(text, registry.all_placeholders())

    total = sum(counts.values())
    audit: Dict[str, object] = {
        "pipeline_step": "entity_blinder",
        "as_of": filing.as_of,
        "counts": dict(sorted(counts.items())),
        "total_redactions": total,
        "placeholder_legend": _placeholder_legend(registry, terms_by_cat),
        "residual_risk": residual,
    }

    return BlindedText(
        as_of=filing.as_of,
        form=filing.form,
        filed=filing.filed,
        text=text,
        audit=audit,
    )


def _placeholder_legend(
    registry: _PlaceholderRegistry, terms_by_cat: Dict[str, List[str]]
) -> Dict[str, str]:
    """Map each assigned placeholder -> the category it redacts (NOT the real
    name — the legend must not re-leak the issuer; it only tells the verifier
    which category a placeholder belongs to).
    """
    legend: Dict[str, str] = {}
    for category, cat_map in registry._maps.items():  # noqa: SLF001
        for placeholder in cat_map.values():
            legend[placeholder] = category
    return dict(sorted(legend.items()))
