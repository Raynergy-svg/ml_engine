"""Real pinned live timers; synthetic admitted observations; no providers."""
import asyncio
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import pytest

from axiom2.nautilus_runtime import CandidateJournal, CandidateObservation, CandidateState, ResearchResult
from axiom2.nautilus_runtime import runtime as module


async def until(condition, timeout=3):
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(.01)


def admitted(now, *, deadline, observation_id='injected'):
    return CandidateObservation('continuous-candidate', 'v1', observation_id, now, deadline,
        hashlib.sha256(b'evidence').hexdigest(), hashlib.sha256(observation_id.encode()).hexdigest(),
        confirmation=True, facts={'source': 'synthetic-injected-admitted'})


def test_native_live_timer_revokes_ready_without_virtual_clock_advance(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'monitor.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            now = runtime.timestamp_ns()
            deadline = now + 180_000_000
            runtime.publish(admitted(now, deadline=deadline), generation=runtime.generation)
            wakeup, = runtime.pending(generation=runtime.generation)
            result = ResearchResult(wakeup.wakeup_id, 'continuous-candidate', 'v1',
                now, deadline, hashlib.sha256(b'research').hexdigest(), True)
            runtime.complete(result, generation=runtime.generation)
            assert journal.snapshot('continuous-candidate', 'v1').state is CandidateState.READY
            await until(lambda: journal.snapshot('continuous-candidate', 'v1').state is CandidateState.EXPIRED)
            status = runtime.status()
            assert status['timer_callbacks'] >= 2
            assert status['runtime'] == 'RUNNING'
            assert status['candidate_ready'] is False
            assert status['data_fresh'] is False
            assert status['live_feed_verified'] is False
            assert status['execution_enabled'] is False
            assert status['capital_authorized'] is False
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_stale_and_future_injected_data_never_qualify(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'stale.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            now = runtime.timestamp_ns()
            assert runtime.publish(admitted(now-2_000_000, deadline=now-1_000_000), generation=runtime.generation) is None
            assert journal.snapshot('continuous-candidate', 'v1').state is CandidateState.EXPIRED
            assert runtime.pending(generation=runtime.generation) == ()
            assert runtime.status()['data_fresh'] is False
            future = admitted(now+10_000_000_000, deadline=now+11_000_000_000, observation_id='future')
            with pytest.raises(ValueError, match='future'):
                runtime.publish(future, generation=runtime.generation)
            assert journal.raw_observation('future') is None
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_restart_preserves_idempotent_results_and_rejects_old_generation(tmp_path):
    async def scenario():
        path = tmp_path / 'idempotent.sqlite'
        journal = CandidateJournal(path)
        first = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        await first.start()
        now = first.timestamp_ns()
        first.publish(admitted(now, deadline=now+5_000_000_000), generation=first.generation)
        wakeup, = first.pending(generation=first.generation)
        result = ResearchResult(wakeup.wakeup_id, 'continuous-candidate', 'v1', now,
            now+5_000_000_000, hashlib.sha256(b'research').hexdigest(), True)
        assert first.complete(result, generation=first.generation).duplicate is False
        assert first.complete(result, generation=first.generation).duplicate is True
        old_generation = first.generation
        await first.shutdown()
        journal.close()
        journal = CandidateJournal(path)
        second = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await second.start()
            assert second.generation == old_generation + 1
            with pytest.raises(RuntimeError, match='generation'):
                second.complete(result, generation=old_generation)
            assert second.complete(result, generation=second.generation).duplicate is True
            assert second.status()['candidate_ready'] is False  # No new admitted data after restart.
        finally:
            await second.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_generation_change_fences_existing_journal_transactions(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'fence.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            generation = runtime.generation
            # Simulate handover before a delayed old callback enters its transaction.
            runtime._connection.execute('UPDATE continuous_runtime SET generation=generation+1')
            now = runtime.timestamp_ns()
            with pytest.raises(RuntimeError, match='generation'):
                runtime.monitor.observe(admitted(now, deadline=now+1_000_000_000))
            assert journal.raw_observation('injected') is None
            assert runtime.status()['runtime'] == 'UNVERIFIED'
            assert runtime.status()['candidate_ready'] is False
            assert runtime._row()['generation'] == generation + 1
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_frozen_host_dispatch_reports_stale_and_stops_fail_closed(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'heartbeat.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            time.sleep(.6)  # Deliberately freeze only the host, not the live clock.
            assert runtime.status()['runtime'] == 'HEARTBEAT_STALE'
            assert runtime.status()['candidate_ready'] is False
            with pytest.raises(RuntimeError, match='generation'):
                runtime.pending(generation=runtime.generation)
            await until(lambda: runtime.status()['runtime'] == 'FAULTED')
            assert runtime.status()['error']
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_sigkill_recovers_pending_wakeup_with_new_fenced_generation(tmp_path):
    path = tmp_path / 'killed.sqlite'
    code = '''
import asyncio,json,sys
from axiom2.nautilus_runtime import CandidateJournal
from axiom2.nautilus_runtime.runtime import NautilusContinuousRuntime
from tests.axiom2.nautilus_runtime.test_continuous_runtime import admitted
def deny(event,args):
    if event in ('socket.connect','socket.bind'):raise RuntimeError('network forbidden')
sys.addaudithook(deny)
async def main():
    journal=CandidateJournal(sys.argv[1]);runtime=NautilusContinuousRuntime(journal,interval_ns=20000000)
    await runtime.start();now=runtime.timestamp_ns()
    runtime.publish(admitted(now,deadline=now+10000000000),generation=runtime.generation)
    print('AXIOM_STATUS '+json.dumps(runtime.status()),flush=True)
    await asyncio.sleep(10)
asyncio.run(main())
'''
    process = subprocess.Popen([sys.executable, '-u', '-c', code, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line.startswith('AXIOM_STATUS '):
                before = json.loads(line.removeprefix('AXIOM_STATUS '))
                break
            if process.poll() is not None:
                pytest.fail(process.stderr.read())
        else:
            pytest.fail('native child did not reach running status')
        assert before['runtime'] == 'RUNNING'
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=3)
        time.sleep(.6)
        async def recovery():
            journal = CandidateJournal(path)
            runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
            assert runtime.status()['runtime'] == 'HEARTBEAT_STALE'
            assert runtime.status()['candidate_ready'] is False
            try:
                await runtime.start()
                assert runtime.generation == before['generation'] + 1
                wakeup, = runtime.pending(generation=runtime.generation)
                now = runtime.timestamp_ns()
                result = ResearchResult(wakeup.wakeup_id, 'continuous-candidate', 'v1', now,
                    now+1_000_000_000, hashlib.sha256(b'research').hexdigest(), True)
                with pytest.raises(RuntimeError, match='generation'):
                    runtime.complete(result, generation=before['generation'])
                assert runtime.complete(result, generation=runtime.generation).duplicate is False
                assert runtime.complete(result, generation=runtime.generation).duplicate is True
                assert len(journal.state_history('continuous-candidate', 'v1')) >= 3
            finally:
                await runtime.shutdown()
                journal.close()
        asyncio.run(recovery())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stdout.close()
        process.stderr.close()


def test_active_lease_denies_second_worker_without_native_node_creation(tmp_path):
    async def scenario():
        path = tmp_path / 'exclusive.sqlite'
        first_journal, second_journal = CandidateJournal(path), CandidateJournal(path)
        first = module.NautilusContinuousRuntime(first_journal, interval_ns=20_000_000)
        second = module.NautilusContinuousRuntime(second_journal, interval_ns=20_000_000)
        try:
            await first.start()
            with pytest.raises(RuntimeError, match='holds the runtime lease'):
                await second.start()
            assert second.generation == 0
            assert second.status()['runtime'] == 'UNVERIFIED'
            assert first.status()['runtime'] == 'RUNNING'
            assert first.status()['generation'] == 1
        finally:
            await first.shutdown()
            first_journal.close()
            second_journal.close()
    asyncio.run(scenario())


def test_future_research_result_cannot_promote_ready(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'future-result.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            now = runtime.timestamp_ns()
            runtime.publish(admitted(now, deadline=now+2_000_000_000), generation=runtime.generation)
            wakeup, = runtime.pending(generation=runtime.generation)
            future = ResearchResult(wakeup.wakeup_id, 'continuous-candidate', 'v1',
                now+1_000_000_000, now+2_000_000_000, hashlib.sha256(b'research').hexdigest(), True)
            with pytest.raises(ValueError, match='future research'):
                runtime.complete(future, generation=runtime.generation)
            assert journal.snapshot('continuous-candidate', 'v1').state is CandidateState.TRIGGERED
            assert len(runtime.pending(generation=runtime.generation)) == 1
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_native_startup_rejection_retains_fault_status(tmp_path):
    async def scenario():
        journals = [CandidateJournal(tmp_path / (name+'.sqlite')) for name in ('active', 'rejected')]
        first, rejected = [module.NautilusContinuousRuntime(j, interval_ns=20_000_000) for j in journals]
        try:
            await first.start()
            with pytest.raises(RuntimeError, match='LiveNode already exists'):
                await rejected.start()
            assert rejected.status()['runtime'] == 'FAULTED'
            assert rejected.status()['error'] == 'RuntimeError'
            assert first.status()['runtime'] == 'RUNNING'
        finally:
            await first.shutdown()
            for journal in journals:
                journal.close()
    asyncio.run(scenario())


def test_clock_regression_retains_root_fault_and_denies_readiness(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'regression.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await runtime.start()
            journal.connection.execute('UPDATE replay_clock SET now_ns=?', (runtime.timestamp_ns()+10_000_000_000,))
            await until(lambda: runtime.status()['runtime'] == 'FAULTED')
            await until(lambda: runtime._task.done())
            assert runtime.status()['error'] == 'CandidateJournalError'
            assert runtime.status()['candidate_ready'] is False
            assert runtime.status()['data_fresh'] is False
        finally:
            await runtime.shutdown()
            journal.close()
    asyncio.run(scenario())


def test_cancelled_native_driver_shutdown_still_releases_node(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'cancel.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        await runtime.start()
        handle = runtime._node.handle()
        runtime._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runtime._task
        await runtime.shutdown()
        assert runtime.status()['runtime'] == 'FAULTED'
        assert handle.is_running is False
        count = runtime.status()['timer_callbacks']
        await asyncio.sleep(.06)
        assert runtime.status()['timer_callbacks'] == count
        journal.close()
    asyncio.run(scenario())
    # Upstream retains its per-thread node guard while caller-held cancellation
    # tracebacks reference run_async. Release that caller frame before rebuilding.
    gc.collect()
    async def replacement_scenario():
        journal = CandidateJournal(tmp_path / 'replacement.sqlite')
        replacement = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        try:
            await replacement.start()
            assert replacement.status()['runtime'] == 'RUNNING'
        finally:
            await replacement.shutdown()
            journal.close()
    asyncio.run(replacement_scenario())


def test_shutdown_cancels_real_live_timer_and_denies_late_inputs(tmp_path):
    async def scenario():
        journal = CandidateJournal(tmp_path / 'shutdown.sqlite')
        runtime = module.NautilusContinuousRuntime(journal, interval_ns=20_000_000)
        await runtime.start()
        assert runtime.status()['runtime'] == 'RUNNING'
        generation = runtime.generation
        await runtime.shutdown()
        status = runtime.status()
        assert status['runtime'] == 'STOPPED'
        assert status['candidate_ready'] is False
        count = status['timer_callbacks']
        await asyncio.sleep(.06)
        assert runtime.status()['timer_callbacks'] == count
        with pytest.raises(RuntimeError):
            runtime.publish(admitted(runtime.timestamp_ns(), deadline=runtime.timestamp_ns()+100_000_000), generation=generation)
        journal.close()
    asyncio.run(scenario())
