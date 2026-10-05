#!/usr/bin/env python3
"""Bounded client for the authority-owning development research service.

The service resolves snapshot IDs from its own allowlist and injects its existing
registry. This client never accepts data paths, proposals, keys or holdout IDs.
Stdio mode emits one request line, consumes one service reply, then emits it.
"""
import argparse
import json
import socket
import sys

MAX_REPLY=16*1024*1024


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    transport=parser.add_mutually_exclusive_group(required=True)
    transport.add_argument('--service-socket')
    transport.add_argument('--service-stdio',action='store_true')
    parser.add_argument('--snapshot-id',required=True)
    parser.add_argument('--action',choices=('preflight','run'),default='preflight')
    args=parser.parse_args()
    request=json.dumps(dict(action=args.action,snapshot_id=args.snapshot_id),separators=(',',':'))+'\n'
    if args.service_stdio:
        print(request,end='',flush=True)
        raw=sys.stdin.buffer.readline(MAX_REPLY+1)
    else:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(600)
            client.connect(args.service_socket)
            client.sendall(request.encode())
            raw=client.makefile('rb').readline(MAX_REPLY+1)
    if not raw or len(raw)>MAX_REPLY or not raw.endswith(b'\n'):
        raise SystemExit('service response is absent, incomplete or too large')
    response=json.loads(raw)
    if response.get('execution_enabled',False) or response.get('holdout_accessed',False):
        raise SystemExit('service response violates development boundary')
    print(json.dumps(response,separators=(',',':'),allow_nan=False))
    if response.get('status') not in ('ADMITTED','COMPLETE','STOPPED'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
