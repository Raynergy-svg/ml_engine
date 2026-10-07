#!/usr/bin/env python3
"""Bounded private observation child preparation; production binding disabled.

Only offline fixtures are accepted until the existing owner's source-admission
and status publisher contracts are verified. Inherited owner pipes are the sole
transport. No socket, provider, model, credential or execution client is created.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys

from axiom2.nautilus_runtime import CandidateJournal, CandidateObservation, ResearchResult
from axiom2.nautilus_runtime.runtime import NautilusContinuousRuntime
from axiom2.nautilus_runtime.journal import CandidateJournalError

MAX_RECORD_BYTES = 65_536
MAX_STORAGE_BYTES = 64 * 1024 * 1024


def private_namespace(evidence_root):
    root = Path(evidence_root).absolute() / 'observation_runtime'
    # The configured evidence root must already exist; never accept paths from
    # records. Reject symlink components and foreign/private-scope mismatches.
    for parent in [root.parent, *root.parent.parents]:
        if parent.is_symlink():
            raise ValueError('unsafe storage namespace: symlink')
    if not root.parent.is_dir():
        raise ValueError('configured evidence root is missing')
    parent_mode = root.parent.stat()
    if parent_mode.st_uid != os.getuid() or stat.S_IMODE(parent_mode.st_mode) & 0o022:
        raise ValueError('unsafe storage namespace: evidence root must be owner-controlled')
    if not root.exists() and not root.is_symlink():
        root.mkdir(mode=0o700)
    for path in [root, *(root / name for name in ('candidates.sqlite', 'candidates.sqlite-wal', 'candidates.sqlite-shm'))]:
        if not path.exists() and not path.is_symlink():
            continue
        mode = path.lstat()
        expected = stat.S_ISDIR(mode.st_mode) if path == root else stat.S_ISREG(mode.st_mode)
        if (not expected or mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077
                or (path != root and mode.st_nlink != 1)):
            raise ValueError('unsafe storage namespace: owner-only directory/files required')
    return root / 'candidates.sqlite'


def record(line):
    if len(line) > MAX_RECORD_BYTES or not line.endswith(b'\n'):
        raise ValueError('record exceeds 64KiB or lacks newline')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate record key')
            result[key] = value
        return result
    payload = json.loads(line.decode('utf-8'), object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(payload, dict):
        raise ValueError('record must be an object')
    return payload


def dispatch(runtime, payload):
    action = payload.get('action')
    fields = {'status': {'action'}, 'shutdown': {'action'},
              'observe': {'action', 'generation', 'observation'},
              'pending': {'action', 'generation'},
              'complete': {'action', 'generation', 'result'}}
    if action not in fields or set(payload) != fields[action]:
        raise ValueError('unsupported action or fields')
    if action == 'observe':
        runtime.publish(CandidateObservation(**payload['observation']), generation=payload['generation'])
    elif action == 'complete':
        runtime.complete(ResearchResult(**payload['result']), generation=payload['generation'])
    elif action == 'pending':
        wakeups = runtime.pending(generation=payload['generation'])
        return dict(status='ACCEPTED', wakeups=[item.to_payload() for item in wakeups])
    return dict(status='ACCEPTED', observation_runtime=runtime.status())


async def serve(path, replies):
    journal = CandidateJournal(path, max_storage_bytes=MAX_STORAGE_BYTES)
    runtime = NautilusContinuousRuntime(journal)
    def reply(value):
        value['execution_enabled'] = value['capital_authorized'] = False
        # Responses are bounded too; never spill a large outbox onto the pipe.
        data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(data) > MAX_RECORD_BYTES:
            data = b'{"status":"REJECTED","error":"response exceeds 64KiB","execution_enabled":false,"capital_authorized":false}\n'
        replies.write(data)
        replies.flush()
    try:
        await runtime.start()
        reply(dict(status='ACCEPTED', observation_runtime=runtime.status()))
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline, MAX_RECORD_BYTES + 1)
            if not line:
                break
            try:
                payload = record(line)
                response = dispatch(runtime, payload)
                if payload['action'] == 'shutdown':
                    await runtime.shutdown()
                    response['observation_runtime'] = runtime.status()
                    reply(response)
                    break
                reply(response)
            except (CandidateJournalError, sqlite3.Error) as exc:
                runtime._fault(type(exc).__name__)
                reply(dict(status='REJECTED', error=type(exc).__name__, observation_runtime=runtime.status()))
                break
            except (ValueError, TypeError, RuntimeError) as exc:
                reply(dict(status='REJECTED', error=str(exc), observation_runtime=runtime.status()))
                if (runtime.status()['runtime'] == 'FAULTED' or len(line) > MAX_RECORD_BYTES
                        or not line.endswith(b'\n')):
                    break
            except Exception as exc:
                runtime._fault(type(exc).__name__)
                reply(dict(status='REJECTED', error=type(exc).__name__, observation_runtime=runtime.status()))
                break
    finally:
        try:
            if runtime._started:
                await runtime.shutdown()
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--offline-fixture', action='store_true')
    args = parser.parse_args()
    if not args.offline_fixture:
        parser.error('production activation is not configured; offline fixtures only')
    allowed_environment = {key: value for key, value in os.environ.items()
                           if key in ('PATH', 'PYTHONPATH', 'LANG', 'LC_ALL', 'TZ')}
    os.environ.clear()
    os.environ.update(allowed_environment)
    os.umask(0o077)
    # Preserve the owner reply pipe before redirecting native C/Rust log output.
    with os.fdopen(os.dup(sys.stdout.fileno()), 'wb') as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        asyncio.run(serve(private_namespace(args.evidence_root), replies))


if __name__ == '__main__':
    main()
