"""Local denial receipt; does not authenticate operational deployment evidence."""
import argparse
import json
from pathlib import Path

REQUIRED_EVIDENCE = (
    'signed exact-build and deployment-policy identity',
    'service identities and cross-process secret/filesystem denial receipts',
    'pinned gateway, exact tool allowlist and unknown-tool denial receipts',
    'provider egress, broad MCP forwarding and alternate-gateway denial receipts',
    'revocation, kill, fault injection and restart-reconciliation receipts',
    'independent verifier trust role, signature and artifact byte verification',
)

def verify_boundary(evidence=None):
    # Claimed signatures, fixture flags and booleans cannot establish authority.
    return dict(status='BLOCKED', execution_enabled=False, operational_isolation_verified=False,
                supplied_evidence_accepted=False, required_evidence=list(REQUIRED_EVIDENCE),
                reason='No authenticated operational evidence verifier is configured.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    try:
        result = verify_boundary(json.loads(args.evidence.read_text()) if args.evidence else None)
    except (OSError, ValueError) as exc:
        result = verify_boundary()
        result['input_error'] = str(exc)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
