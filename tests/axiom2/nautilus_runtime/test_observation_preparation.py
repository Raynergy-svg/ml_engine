"""Private, offline preparation; real SQLite and pinned native child process."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
import time

import pytest

from axiom2.nautilus_runtime import CandidateJournal, CandidateObservation, ResearchResult
from axiom2.nautilus_runtime.runtime import NautilusContinuousRuntime

WORKER = Path(__file__).resolve().parents[3] / 'scripts/axiom2_nautilus_observation_worker.py'


def test_bounded_journal_denies_growth_and_retains_committed_records(tmp_path):
    journal = CandidateJournal(tmp_path / 'candidates.sqlite', max_storage_bytes=512 * 1024)
    journal.connection.execute('CREATE TABLE quota_fixture (payload BLOB)')
    journal.check_storage()
    journal.connection.execute('INSERT INTO quota_fixture VALUES (?)', (b'kept',))
    with pytest.raises(sqlite3.DatabaseError, match='database or disk is full'):
        with journal._transaction():
            journal.connection.execute('INSERT INTO quota_fixture VALUES (?)', (b'x' * 512_000,))
    assert journal.connection.execute('SELECT payload FROM quota_fixture').fetchall()[0][0] == b'kept'
    assert journal.storage_bytes() <= 512 * 1024
    journal.close()


def test_reader_pinning_wal_denies_new_transaction(tmp_path):
    path = tmp_path / 'candidates.sqlite'
    journal = CandidateJournal(path, max_storage_bytes=512 * 1024)
    journal.check_storage()
    reader = sqlite3.connect(path)
    reader.execute('BEGIN')
    reader.execute('SELECT * FROM replay_clock').fetchall()
    journal.connection.execute('UPDATE replay_clock SET now_ns=1')
    try:
        with pytest.raises(RuntimeError, match='checkpoint'):
            journal.advance_time(2)
        assert journal.now_ns == 1
        assert journal.storage_bytes() <= 512 * 1024
    finally:
        reader.close()
        journal.close()


def test_existing_oversize_store_is_rejected_before_mutation(tmp_path):
    path = tmp_path / 'candidates.sqlite'
    original = b'kept' * 40_000
    path.write_bytes(original)
    with pytest.raises(RuntimeError, match='storage allocation'):
        CandidateJournal(path, max_storage_bytes=512 * 1024)
    assert path.read_bytes() == original
    assert not path.with_name(path.name+'-wal').exists()


def test_unwritable_journal_stops_actual_native_handle_and_clears_ready(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'candidates.sqlite', max_storage_bytes=512 * 1024)
        runtime = NautilusContinuousRuntime(journal)
        await runtime.start()
        handle = runtime._node.handle()
        clock = runtime._actor.clock
        journal.close()  # Real SQLite failure, not a mocked native stop.
        runtime._fault('storage-unavailable')
        assert runtime.status()['runtime'] == 'FAULTED'
        assert runtime.status()['candidate_ready'] is False
        with pytest.raises(sqlite3.ProgrammingError):
            await runtime.shutdown()
        assert not handle.is_running
        assert clock.timer_names() == []
        assert runtime._node is None
    asyncio.run(scenario())


def test_native_callback_sqlite_full_cannot_leave_ready_or_accept_invalidation(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'candidates.sqlite', max_storage_bytes=1024 * 1024)
        runtime = NautilusContinuousRuntime(journal)
        try:
            await runtime.start()
            now = runtime.timestamp_ns()
            digest = hashlib.sha256(b'fixture').hexdigest()
            observation = CandidateObservation('fixture', 'v1', 'original', now, now+10_000_000_000,
                                               digest, digest, confirmation=True)
            runtime.publish(observation, generation=runtime.generation)
            wakeup, = runtime.pending(generation=runtime.generation)
            runtime.complete(ResearchResult(wakeup.wakeup_id, 'fixture', 'v1', now,
                                           now+10_000_000_000, digest, True), generation=runtime.generation)
            assert runtime.status()['candidate_ready'] is True
            journal.connection.execute('CREATE TABLE quota_fixture(payload BLOB)')
            for _ in range(1000):
                try:
                    with journal._transaction():
                        journal.connection.execute('INSERT INTO quota_fixture VALUES (?)', (b'x'*1000,))
                except sqlite3.DatabaseError:
                    break
            else:
                pytest.fail('fixture did not reach the real SQLite page limit')
            invalidation = CandidateObservation('fixture', 'v1', 'invalidation', now, now+10_000_000_000,
                digest, digest, invalidated=True, invalidation_reason='fixture invalidation',
                facts={'large_record': 'x'*20_000})
            with pytest.raises(RuntimeError, match='callback failed closed'):
                runtime.publish(invalidation, generation=runtime.generation)
            assert journal.raw_observation('invalidation') is None
            assert runtime.status()['runtime'] == 'FAULTED'
            assert runtime.status()['candidate_ready'] is False
            assert journal.storage_bytes() <= 1024 * 1024
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def receive(child):
    assert select.select([child.stdout], [], [], 5)[0], 'bounded child response timeout'
    return json.loads(child.stdout.readline())


def send(child, payload):
    child.stdin.write(json.dumps(payload).encode() + b'\n')
    child.stdin.flush()
    return receive(child)


def launch(root, stderr):
    # Explicit test environment: no inherited broker/provider/signer credentials.
    env = {key: os.environ[key] for key in ('PATH', 'PYTHONPATH') if key in os.environ}
    return subprocess.Popen([sys.executable, str(WORKER), '--offline-fixture', '--evidence-root', str(root)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=env)


def test_default_entrypoint_cannot_activate_or_create_state(tmp_path):
    result = subprocess.run([sys.executable, str(WORKER), '--evidence-root', str(tmp_path)],
                            input=b'', capture_output=True, timeout=5)
    assert result.returncode != 0
    assert b'activation is not configured' in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('attack', ['symlink', 'public', 'parent-public', 'hardlink'])
def test_child_rejects_unsafe_storage_namespace(tmp_path, attack):
    root = tmp_path / 'observation_runtime'
    if attack == 'symlink':
        target = tmp_path / 'target'
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    elif attack == 'public':
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    elif attack == 'parent-public':
        tmp_path.chmod(0o777)
    else:
        root.mkdir(mode=0o700)
        original = tmp_path / 'original.sqlite'
        with sqlite3.connect(original) as db:
            db.execute('CREATE TABLE original (payload)')
        os.link(original, root / 'candidates.sqlite')
    with (tmp_path / 'stderr').open('wb') as errors:
        child = launch(tmp_path, errors)
        child.communicate(timeout=5)
        assert child.returncode != 0
    assert b'unsafe storage namespace' in (tmp_path / 'stderr').read_bytes()
    if attack != 'hardlink':
        assert not (root / 'candidates.sqlite').exists()


def test_real_child_idle_timers_ready_expiry_and_restart(tmp_path):
    with (tmp_path / 'stderr').open('wb') as errors:
        child = launch(tmp_path, errors)
        try:
            first = receive(child)['observation_runtime']
            generation = first['generation']
            assert first['runtime'] == 'RUNNING'
            time.sleep(.15)
            later = send(child, {'action': 'status'})['observation_runtime']
            assert later['timer_callbacks'] > first['timer_callbacks']
            now = time.time_ns()
            digest = hashlib.sha256(b'offline-fixture').hexdigest()
            observation = dict(candidate_id='fixture', candidate_version='v1', observation_id='fixture-1',
                observed_at_ns=now, freshness_deadline_ns=now+800_000_000,
                evidence_digest=digest, raw_observation_digest=digest, confirmation=True,
                facts={'source': 'offline-fixture'})
            accepted = send(child, {'action': 'observe', 'generation': generation, 'observation': observation})
            assert accepted['status'] == 'ACCEPTED'
            wakeup, = send(child, {'action': 'pending', 'generation': generation})['wakeups']
            result = dict(wakeup_id=wakeup['wakeup_id'], candidate_id='fixture', candidate_version='v1',
                completed_at_ns=now, fresh_until_ns=now+800_000_000, evidence_digest=digest, qualifies=True)
            assert send(child, {'action': 'complete', 'generation': generation, 'result': result})['status'] == 'ACCEPTED'
            ready = send(child, {'action': 'status'})['observation_runtime']
            assert ready['candidate_ready'] is True
            assert ready['owner_transport_verified'] is False
            assert ready['execution_enabled'] is ready['capital_authorized'] is False
            time.sleep(.85)
            expired = send(child, {'action': 'status'})['observation_runtime']
            assert expired['runtime'] == 'RUNNING' and expired['candidate_ready'] is False
            assert send(child, {'action': 'observe', 'generation': generation-1, 'observation': observation})['status'] == 'REJECTED'
            stopped = send(child, {'action': 'shutdown'})['observation_runtime']
            assert stopped['runtime'] == 'STOPPED'
            child.wait(timeout=5)
            assert child.returncode == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
        replacement = launch(tmp_path, errors)
        try:
            resumed = receive(replacement)['observation_runtime']
            assert resumed['generation'] == generation+1
            assert resumed['data_fresh'] is resumed['candidate_ready'] is False
            assert send(replacement, {'action': 'shutdown'})['observation_runtime']['runtime'] == 'STOPPED'
            replacement.wait(timeout=5)
        finally:
            if replacement.poll() is None:
                replacement.kill()
                replacement.wait(timeout=5)
            replacement.stdin.close()
            replacement.stdout.close()
    folder = tmp_path / 'observation_runtime'
    assert folder.stat().st_mode & 0o777 == 0o700
    assert (folder / 'candidates.sqlite').stat().st_mode & 0o777 == 0o600


def test_oversize_frame_stops_without_candidate_write(tmp_path):
    with (tmp_path / 'stderr').open('wb') as errors:
        child = launch(tmp_path, errors)
        try:
            receive(child)
            child.stdin.write(b'x' * 65536 + b'\n')
            child.stdin.flush()
            assert receive(child)['status'] == 'REJECTED'
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
    with sqlite3.connect(tmp_path / 'observation_runtime/candidates.sqlite') as connection:
        assert connection.execute('SELECT COUNT(*) FROM raw_observations').fetchone()[0] == 0


def test_invalid_records_and_production_controls_cannot_mutate_candidates(tmp_path):
    with (tmp_path / 'stderr').open('wb') as errors:
        child = launch(tmp_path, errors)
        try:
            initial = receive(child)['observation_runtime']
            for line in (b'{"action":"status","action":"observe"}\n',
                         b'{"action":"restart"}\n',
                         b'{"action":"status","credentials":"forbidden"}\n',
                         b'{"action":"status","extra":NaN}\n', b'[]\n', b'\xff\n'):
                child.stdin.write(line)
                child.stdin.flush()
                rejected = receive(child)
                assert rejected['status'] == 'REJECTED'
                assert rejected['execution_enabled'] is rejected['capital_authorized'] is False
            assert send(child, {'action': 'status'})['observation_runtime']['generation'] == initial['generation']
            send(child, {'action': 'shutdown'})
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
    with sqlite3.connect(tmp_path / 'observation_runtime/candidates.sqlite') as connection:
        assert connection.execute('SELECT COUNT(*) FROM raw_observations').fetchone()[0] == 0
