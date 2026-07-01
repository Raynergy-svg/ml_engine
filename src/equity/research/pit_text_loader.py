"""Point-in-time (PIT) primary-document TEXT loader for SEC filings (Track B).

Feeds the entity-blinder + research scorer pre-registered in
``docs/experiment-equity-research-alpha-prereg-2026-06-30.md``. Given an issuer
(ticker + CIK) and an ``as_of`` date, it returns the MOST RECENT SEC filing of a
requested form (default 10-K/10-Q/8-K) whose ``filingDate <= as_of``, as plain
text (primary HTML document stripped). Anything filed after ``as_of`` is
lookahead and is never returned.

The PIT guarantee is enforced in two layers:

1. :func:`select_pit_filing` filters the EDGAR submission list to rows with
   ``filingDate <= as_of`` BEFORE any network fetch of a document, then picks the
   most-recently-filed survivor. A row filed after ``as_of`` can never be chosen.
2. :class:`~src.equity.research.contracts.FilingText` re-asserts ``filed <= as_of``
   in ``__post_init__`` — so even a logic bug here cannot mint a lookahead object;
   it raises ``ValueError`` instead.

DATA SOURCE — SEC EDGAR (free, official, preserves the real filing date):

* Submission list: ``https://data.sec.gov/submissions/CIK##########.json`` (CIK
  zero-padded to 10 digits). The ``filings.recent`` block carries parallel arrays
  (form / filingDate / accessionNumber / primaryDocument / ...). For deep history
  it references older paginated files via ``filings.files[]`` — we fetch those
  additional files when the recent block does not already cover ``as_of``.
* Primary document:
  ``https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primaryDocument}``

EDGAR requires a descriptive User-Agent and a polite rate limit (<= 10 req/s). We
reuse the request discipline of :mod:`src.equity.edgar_fundamentals` (UA header,
explicit (connect, read) timeouts, exponential backoff on 5xx/429/timeout, never
retry other 4xx).

KNOWN LIMITATION (labelled, not hidden): the additional ``filings.files[]`` are
fetched only when the recent block's oldest entry is still newer than ``as_of``
(i.e. we have to page back to find a filing on/before ``as_of``). Very deep
history (pre ~2001 for some filers) may lack a usable ``primaryDocument`` entry in
the index arrays — in that case the loader returns ``None`` and logs a warning
rather than guessing. It NEVER returns a document filed after ``as_of``.

OFFLINE only — network fetch + optional atomic cache write to disk; nothing runs
in a process, trades, unhalts, or touches the hot path.
"""
from __future__ import annotations

import logging
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from src.equity.research.contracts import FilingText

logger = logging.getLogger(__name__)

# SEC requires a descriptive UA with contact info; 10 req/s ceiling.
SEC_UA = "ml_engine-research dcertan84@gmail.com"
_REQUEST_SPACING_S = 0.13          # stay comfortably under 10 req/s
_TIMEOUT = (5, 30)                 # (connect, read)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
_SUBMISSIONS_PRIMARY = "CIK{cik}.json"
_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
)

DEFAULT_FORMS: Tuple[str, ...] = ("10-K", "10-Q", "8-K")


class _TextExtractor(HTMLParser):
    """Stdlib-only HTML -> text: drop script/style, collapse whitespace.

    Dependency-light by design (no bs4 / lxml). Good enough for the blinder /
    scorer, which want readable prose, not a faithful DOM.
    """

    _SKIP_TAGS = {"script", "style", "head", "title"}
    _BLOCK_TAGS = {
        "p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4",
        "h5", "h6", "section", "article", "ul", "ol", "thead", "tbody",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    """Strip HTML/XHTML to collapsed plain text (stdlib ``html.parser`` only).

    Removes script/style, inserts newlines at block boundaries, and collapses
    runs of whitespace. Returns ``""`` for empty input. Pure + offline-testable.
    """
    if not html:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
        raw = parser.get_text()
    except (ValueError, AssertionError) as exc:
        # Malformed markup: fall back to a crude tag strip rather than crash.
        logger.warning("html_to_text parser error (%s); using fallback strip", exc)
        import re

        raw = re.sub(r"<[^>]+>", " ", html)

    # Collapse whitespace: trim each line, drop blank lines, single spaces.
    lines = [" ".join(ln.split()) for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _iter_recent_rows(recent: dict) -> Iterable[dict]:
    """Zip the parallel arrays of a submissions ``recent``/file block into dicts.

    Yields ``{form, filingDate, accessionNumber, primaryDocument}`` per index.
    Tolerates ragged/short arrays by stopping at the shortest required column.
    """
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accs = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    n = min(len(forms), len(dates), len(accs), len(docs))
    for i in range(n):
        yield {
            "form": str(forms[i]),
            "filingDate": str(dates[i]),
            "accessionNumber": str(accs[i]),
            "primaryDocument": str(docs[i]),
        }


def select_pit_filing(
    rows: Iterable[dict],
    as_of: str,
    forms: Sequence[str],
) -> Optional[dict]:
    """Pure selection: most-recent row of an allowed form with filingDate <= as_of.

    ``rows`` are dicts with at least ``form``, ``filingDate``, ``accessionNumber``,
    ``primaryDocument``. ISO dates compare lexicographically, so string ``<=`` is a
    correct date comparison. Rows missing a ``primaryDocument`` are skipped (no
    fetchable text). Returns the winning row dict, or ``None`` if nothing qualifies.

    This is the PIT gate: a row with ``filingDate > as_of`` can never be returned.
    """
    allowed = set(forms)
    best: Optional[dict] = None
    for row in rows:
        form = str(row.get("form", ""))
        if form not in allowed:
            continue
        filed = str(row.get("filingDate", ""))
        if not filed or filed > as_of:        # PIT gate — strict lookahead block
            continue
        if not str(row.get("primaryDocument", "")):
            continue
        if best is None or filed > str(best.get("filingDate", "")):
            best = dict(row)
    return best


def _get(session: requests.Session, url: str, *, retries: int = 4) -> Optional[requests.Response]:
    """GET with UA, explicit timeouts, exponential backoff on 5xx/429/timeout.

    Returns the 200 ``Response``; ``None`` on 404 / other-4xx / exhaustion. Never
    retries non-429 4xx (per the project retry gate). Catches only the specific
    transient exceptions; no bare except.
    """
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": SEC_UA}, timeout=_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.warning(
                "EDGAR transient error %s on %s (attempt %d/%d)",
                exc, url, attempt + 1, retries,
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            logger.warning("EDGAR 404 on %s", url)
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning(
                "EDGAR %d on %s (attempt %d/%d) — backing off",
                resp.status_code, url, attempt + 1, retries,
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        logger.warning("EDGAR %d on %s — not retrying", resp.status_code, url)
        return None
    return None


def _get_json(session: requests.Session, url: str) -> Optional[dict]:
    resp = _get(session, url)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        logger.warning("EDGAR returned non-JSON on %s", url)
        return None


def _candidate_blocks(
    session: requests.Session, submissions: dict, cik10: str, as_of: str,
) -> Iterable[dict]:
    """Yield the ``recent`` block, then older paginated files IF still needed.

    We only page back to ``filings.files[]`` when the recent block's oldest entry
    is newer than ``as_of`` (meaning the on/before-as_of filing predates the recent
    window). Older files whose ``filingTo`` is before ``as_of`` are the relevant
    ones; we fetch in newest-first order so the first hit is the most recent.
    """
    filings = submissions.get("filings", {}) or {}
    recent = filings.get("recent", {}) or {}
    yield recent

    recent_dates = recent.get("filingDate") or []
    oldest_recent = min(recent_dates) if recent_dates else None
    # If the recent block already reaches on/before as_of, no paging needed.
    if oldest_recent is not None and oldest_recent <= as_of:
        return

    files = filings.get("files", []) or []
    # Newest-first: a file is only useful if its window starts on/before as_of.
    usable = [
        f for f in files
        if str(f.get("filingFrom", "")) <= as_of and f.get("name")
    ]
    usable.sort(key=lambda f: str(f.get("filingTo", "")), reverse=True)
    for f in usable:
        time.sleep(_REQUEST_SPACING_S)
        data = _get_json(session, SUBMISSIONS_URL.format(name=f["name"]))
        if isinstance(data, dict):
            yield data


def load_pit_filing(
    ticker: str,
    cik: str,
    as_of: str,
    *,
    forms: Sequence[str] = DEFAULT_FORMS,
    session: Optional[requests.Session] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[FilingText]:
    """Most-recent filing of one of ``forms`` with ``filed <= as_of``, as text.

    Returns a :class:`FilingText` (raw primary-document text, HTML stripped) or
    ``None`` if no qualifying filing exists by ``as_of`` (or EDGAR is unreachable).

    PIT is guaranteed by :func:`select_pit_filing` (filters ``filed <= as_of``
    before any document fetch) AND by ``FilingText.__post_init__`` (re-asserts the
    invariant). ``cik`` may be given in any form; it is zero-padded to 10 digits
    for the submissions endpoint and used as an int for the Archives path.
    """
    sess = session or requests.Session()
    cik_digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not cik_digits:
        logger.warning("load_pit_filing: non-numeric CIK %r for %s", cik, ticker)
        return None
    cik10 = f"{int(cik_digits):010d}"
    cik_int = int(cik_digits)

    sub_url = SUBMISSIONS_URL.format(name=_SUBMISSIONS_PRIMARY.format(cik=cik10))
    submissions = _get_json(sess, sub_url)
    if not submissions:
        logger.warning(
            "load_pit_filing: no submissions JSON for %s (CIK %s)", ticker, cik10
        )
        return None

    # Walk recent + (lazily) older blocks; keep the best PIT row across blocks.
    best: Optional[dict] = None
    for block in _candidate_blocks(sess, submissions, cik10, as_of):
        cand = select_pit_filing(_iter_recent_rows(block), as_of, forms)
        if cand is not None and (
            best is None or cand["filingDate"] > best["filingDate"]
        ):
            best = cand
    if best is None:
        logger.warning(
            "load_pit_filing: no %s filing for %s with filed <= %s",
            tuple(forms), ticker, as_of,
        )
        return None

    acc_nodash = best["accessionNumber"].replace("-", "")
    doc_url = _ARCHIVES_URL.format(
        cik_int=cik_int, acc_nodash=acc_nodash, doc=best["primaryDocument"]
    )

    cached = _read_cache(cache_dir, cik10, acc_nodash, best["primaryDocument"])
    if cached is not None:
        text = cached
    else:
        time.sleep(_REQUEST_SPACING_S)
        resp = _get(sess, doc_url)
        if resp is None:
            logger.warning(
                "load_pit_filing: failed to fetch primary doc %s for %s",
                doc_url, ticker,
            )
            return None
        text = html_to_text(resp.text)
        if not text:
            logger.warning(
                "load_pit_filing: empty text after strip for %s (%s)",
                ticker, doc_url,
            )
            return None
        _write_cache(cache_dir, cik10, acc_nodash, best["primaryDocument"], text)

    # FilingText re-asserts filed <= as_of; a logic slip here raises, not lies.
    return FilingText(
        ticker=str(ticker),
        cik=cik10,
        as_of=str(as_of),
        form=best["form"],
        filed=best["filingDate"],
        text=text,
    )


def load_pit_filings(
    items: Iterable[Tuple[str, str, str]],
    *,
    forms: Sequence[str] = DEFAULT_FORMS,
    session: Optional[requests.Session] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[Tuple[str, str, str], Optional[FilingText]]:
    """Batch helper over ``(ticker, cik, as_of)`` tuples — reuses one session.

    Returns ``{(ticker, cik, as_of): FilingText | None}``. A failure for one item
    is recorded as ``None`` (logged), never aborting the whole batch.
    """
    sess = session or requests.Session()
    out: Dict[Tuple[str, str, str], Optional[FilingText]] = {}
    for ticker, cik, as_of in items:
        out[(ticker, cik, as_of)] = load_pit_filing(
            ticker, cik, as_of, forms=forms, session=sess, cache_dir=cache_dir
        )
    return out


# --------------------------------------------------------------------------- #
# Cache (optional, atomic). Keyed by CIK/accession/document — immutable on EDGAR.
# --------------------------------------------------------------------------- #
def _cache_path(
    cache_dir: Optional[Path], cik10: str, acc_nodash: str, doc: str
) -> Optional[Path]:
    if cache_dir is None:
        return None
    safe_doc = doc.replace("/", "_")
    return Path(cache_dir) / cik10 / f"{acc_nodash}__{safe_doc}.txt"


def _read_cache(
    cache_dir: Optional[Path], cik10: str, acc_nodash: str, doc: str
) -> Optional[str]:
    path = _cache_path(cache_dir, cik10, acc_nodash, doc)
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("cache read failed for %s: %s", path, exc)
        return None
    return text or None


def _write_cache(
    cache_dir: Optional[Path], cik10: str, acc_nodash: str, doc: str, text: str
) -> None:
    path = _cache_path(cache_dir, cik10, acc_nodash, doc)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)        # atomic
    except OSError as exc:
        logger.warning("cache write failed for %s: %s", path, exc)
