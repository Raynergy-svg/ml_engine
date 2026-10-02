"""Offline official read-contract verification. Never calls the connector.

Live verification belongs to the authorized protected service. This script does
not accept credentials, account identifiers or raw response files. PASS means
only declaration pin integrity and importability; Task13 live gates stay closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.axiom2.brokers.robinhood_readonly import DECLARED_CONTRACT_SHA256
from src.axiom2.brokers.contracts import CapabilitySnapshot

TOOLS=('get_accounts','get_portfolio','get_equity_positions','get_equity_orders','get_equity_quotes','get_equity_tradability')

def verify_contracts(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:raise ValueError('INVALID_CONTRACT_ARTIFACT')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DECLARED_CONTRACT_SHA256:raise ValueError('PROVIDER_SCHEMA_DRIFT')
    data=json.loads(raw)
    if data['format']!='official-robinhood-mcp-tool-declarations-v1' or {entry['tool'] for entry in data['entries']}!={'mcp__codex_apps__robinhood_trading_'+name for name in TOOLS}:raise ValueError('EXACT_READ_TOOL_INVENTORY_REQUIRED')
    return dict(status='PASS',scope='offline-declared-contract-integrity-only',
        live_connection_verified=False,live_nonempty_or_pagination_verified=False,
        broker_actions=0,operational_proof=False,
        capability=CapabilitySnapshot(declared_contract_sha256=DECLARED_CONTRACT_SHA256).model_dump(mode='json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contracts',type=Path,default=ROOT/'config/axiom2/robinhood_readonly_contracts.json')
    args=parser.parse_args()
    try:result=verify_contracts(args.contracts)
    except (ValueError,OSError):
        print(json.dumps(dict(status='BLOCKED',reason='INVALID_OR_CHANGED_DECLARED_CONTRACT',execution_enabled=False,capital_authorized=False)))
        return 2
    print(json.dumps(result,sort_keys=True,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
