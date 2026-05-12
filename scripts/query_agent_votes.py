#!/usr/bin/env python3
"""Query the agent-vote audit index (Bucket D2).

Examples::

    # Every trade in the last 30 days where mean_reversion voted NO and we lost.
    python scripts/query_agent_votes.py --agent mean_reversion --passed 0 \\
        --outcome loss --since 30d

    # Find loss reasons mentioning "drawdown".
    python scripts/query_agent_votes.py --outcome loss --fts drawdown

    # Show the 10 most recent votes from the trend agent across all regimes.
    python scripts/query_agent_votes.py --agent trend --limit 10

Bootstrap (first time)::

    python scripts/query_agent_votes.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.tui.agent_vote_index import AgentVoteIndex, DB_PATH, JOURNAL_PATH  # noqa: E402


def _parse_since(value: str) -> str:
    """Accept either an ISO timestamp or a duration like '30d' / '12h' / '90m'.

    Duration strings convert to (now - delta) in UTC ISO format. ISO strings
    pass through unchanged.
    """
    v = value.strip()
    if not v:
        return ""
    suffix_to_kwarg = {"d": "days", "h": "hours", "m": "minutes"}
    if len(v) >= 2 and v[-1] in suffix_to_kwarg:
        try:
            n = int(v[:-1])
        except ValueError:
            return v
        kwargs = {suffix_to_kwarg[v[-1]]: n}
        cutoff = datetime.now(timezone.utc) - timedelta(**kwargs)
        return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    return v


def _format_table(rows: List[dict]) -> str:
    if not rows:
        return "(no matches)"
    headers = ["trade_id", "ts", "agent", "passed", "score", "weight", "outcome", "pnl", "reason"]
    out = [
        f"{headers[0]:<10} {headers[1]:<24} {headers[2]:<22} "
        f"{headers[3]:>6} {headers[4]:>6} {headers[5]:>7} "
        f"{headers[6]:<10} {headers[7]:>8}  {headers[8]}"
    ]
    out.append("-" * 130)
    for r in rows:
        ts = (r.get("closed_ts_iso") or "")[:23]
        score = r.get("score")
        weight = r.get("weight")
        pnl = r.get("pnl_usd")
        out.append(
            f"{(r.get('trade_id') or '')[:10]:<10} "
            f"{ts:<24} "
            f"{(r.get('agent_name') or '')[:22]:<22} "
            f"{int(r.get('passed') or 0):>6} "
            f"{(f'{score:.2f}' if score is not None else '?'):>6} "
            f"{(f'{weight:.2f}' if weight is not None else '?'):>7} "
            f"{(r.get('outcome') or '')[:10]:<10} "
            f"{(f'{pnl:>+7.2f}' if pnl is not None else '       ?'):>8}  "
            f"{(r.get('reason_text') or '')[:60]}"
        )
    return "\n".join(out)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query agent-vote audit index.")
    p.add_argument("--agent", help="Exact agent name (e.g. mean_reversion, trend).")
    p.add_argument("--passed", type=int, choices=(0, 1),
                   help="0 = vote was NO/failed, 1 = vote was YES/passed.")
    p.add_argument("--outcome", choices=("win", "loss", "breakeven", "open"),
                   help="Trade outcome filter.")
    p.add_argument("--regime", choices=("LOW", "NORMAL", "HIGH", "EXTREME"),
                   help="Volatility regime filter.")
    p.add_argument("--since", help="ISO timestamp or duration suffix (30d, 12h, 90m).")
    p.add_argument("--fts", dest="reason_fts",
                   help="FTS5 search against reason_text (partial input ok).")
    p.add_argument("--limit", type=int, default=50, help="Max rows (default: 50).")
    p.add_argument("--db", type=Path, default=DB_PATH, help="Path to agent_vote_index.db.")
    p.add_argument("--journal", type=Path, default=JOURNAL_PATH,
                   help="Path to trade_journal_rl.json (for --rebuild / --sync).")
    p.add_argument("--rebuild", action="store_true",
                   help="Truncate and re-ingest the full journal before querying.")
    p.add_argument("--sync", action="store_true",
                   help="Run incremental sync before querying (cheap, watermark-driven).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        index = AgentVoteIndex.open(db_path=args.db)
    except Exception as e:  # noqa: BLE001 — surface DB-open failures
        print(f"Failed to open index DB at {args.db}: {e}")
        return 2

    try:
        if args.rebuild:
            n = index.rebuild_full(journal_path=args.journal)
            print(f"Rebuilt index from {n} journal entries.")
        elif args.sync:
            stats = index.sync_from_journal(journal_path=args.journal)
            print(f"Synced: {stats}")

        since_norm = _parse_since(args.since) if args.since else None

        rows = index.query_votes(
            agent_name=args.agent,
            passed=args.passed,
            outcome=args.outcome,
            regime=args.regime,
            since=since_norm,
            reason_fts=args.reason_fts,
            limit=args.limit,
        )
        print(_format_table(rows))
        print(f"\n{len(rows)} row(s).")
        return 0
    finally:
        index.close()


if __name__ == "__main__":
    sys.exit(main())
