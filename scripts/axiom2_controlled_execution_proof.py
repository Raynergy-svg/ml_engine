"""Disabled Task18 preparation; contains no broker or approval capability."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

FIELDS = {'account_alias', 'instrument_id', 'side', 'quantity', 'limit_price_cents',
          'capital_cap_cents', 'expires_at', 'kill_plan'}
REQUIRED_READINESS = (
    'authenticated genuine holdout promotion and exact promoted artifacts',
    'completed live shadow qualification and fresh complete reconciliation',
    'authenticated Task17 operational denial receipt for exact build',
    'separately approved build and deployment',
    'provider-confirmed dedicated account allowlist and trade approvals ON',
    'signed bounded operator authorization for exact intent, account, total cap and expiry',
    'reviewed risk intent, regular session, cash, inventory and cost reservation',
)

def prepare_proof(proposal, readiness=None):
    if type(proposal) is not dict or set(proposal) != FIELDS:
        raise ValueError('Exact preparation fields required.')
    for field in ('account_alias', 'instrument_id', 'kill_plan'):
        if type(proposal[field]) is not str or not proposal[field].strip():
            raise ValueError('Nonempty ' + field + ' required.')
    if type(proposal['side']) is not str or proposal['side'] != 'BUY':
        raise ValueError('Disabled preparation supports a long-only BUY example.')
    for field in ('quantity', 'limit_price_cents', 'capital_cap_cents'):
        if type(proposal[field]) is not int or proposal[field] <= 0:
            raise ValueError('Positive integer ' + field + ' required.')
    if proposal['quantity'] * proposal['limit_price_cents'] > proposal['capital_cap_cents']:
        raise ValueError('Order principal exceeds declared capital cap.')
    if type(proposal['expires_at']) is not str or not proposal['expires_at'].endswith('Z'):
        raise ValueError('Explicit UTC expiry required.')
    try:
        expiry = datetime.fromisoformat(proposal['expires_at'][:-1] + '+00:00')
        if 'T' not in proposal['expires_at'] or expiry.utcoffset() != timedelta(0):
            raise ValueError('UTC date and time required.')
    except ValueError:
        raise ValueError('Valid UTC expiry required.') from None
    return dict(status='BLOCKED', execution_enabled=False, capital_authorized=False,
                broker_actions_created=[], proposal=dict(proposal), readiness_verified=False,
                required_readiness=list(REQUIRED_READINESS),
                reason='HARD HUMAN GATE unexecuted; no authenticated readiness verifier or gateway.',
                limitations=['Principal cap excludes fees; this is not a risk approval.',
                             'Expiry is descriptive; authoritative current-time validation remains required.',
                             'No broker review, approval, submission or cancellation is requested.'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--readiness', type=Path)
    args = parser.parse_args()
    try:
        result = prepare_proof(json.loads(args.proposal.read_text()),
            json.loads(args.readiness.read_text()) if args.readiness else None)
    except (OSError, ValueError) as exc:
        result = dict(status='BLOCKED', execution_enabled=False, capital_authorized=False, broker_actions_created=[], input_error=str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
