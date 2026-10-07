"""Public-interface Nautilus replay runtime for Axiom candidate observations."""

from __future__ import annotations

from typing import Any
import asyncio
import time

from .contracts import CandidateObservation, MaterialEvent
from .journal import CandidateJournal
from .policy import CandidateMonitor


class NautilusRuntimeUnavailable(RuntimeError):
    """Raised when the pinned Nautilus runtime is not installed."""


class NautilusReplayRuntime:
    """Deliver admitted observations through Nautilus into Axiom policy.

    This is deliberately replay-only. It uses Nautilus Clock and MessageBus
    mechanics, while its start/stop methods are adapter lifecycle gates. It
    exposes no order or execution client methods and does not create a
    TradingNode, broker, or credentialed client.
    """

    TOPIC = "axiom2.nautilus.admitted_observation"

    def __init__(
        self,
        monitor: CandidateMonitor,
        journal: CandidateJournal,
        *,
        trader_id: str = "TRADER-AXIOM2-REPLAY",
        topic: str = TOPIC,
    ) -> None:
        try:
            from nautilus_trader.common import Clock, MessageBus
            from nautilus_trader.model import TraderId
        except ModuleNotFoundError as exc:
            raise NautilusRuntimeUnavailable(
                "the pinned Nautilus package is required for replay runtime"
            ) from exc

        self.monitor = monitor
        self.journal = journal
        self.topic = topic
        self.clock = Clock.new_test()
        self.clock.set_time(journal.now_ns)
        self.bus = MessageBus(TraderId(trader_id), clock=self.clock)
        self._started = False
        self._disposed = False
        self._last_material_event: MaterialEvent | None = None
        self.bus.subscribe(self.topic, self._on_message)

    @property
    def is_running(self) -> bool:
        return self._started and not self._disposed

    @property
    def last_material_event(self) -> MaterialEvent | None:
        return self._last_material_event

    def start(self) -> None:
        if self._disposed:
            raise RuntimeError("runtime is disposed")
        if not self._started:
            self._started = True

    def stop(self) -> None:
        if self._disposed:
            return
        if self._started:
            self._started = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self.stop()
        self.bus.dispose()
        self._disposed = True

    def set_time(self, to_time_ns: int) -> None:
        if not self.is_running:
            raise RuntimeError("runtime must be running")
        self.journal.advance_time(to_time_ns)
        self.clock.set_time(to_time_ns)

    def publish(self, observation: CandidateObservation) -> MaterialEvent | None:
        if not self.is_running:
            raise RuntimeError("runtime must be running")
        self._last_material_event = None
        self.bus.publish(
            self.topic,
            observation.to_payload(),
            external_pub=False,
        )
        return self._last_material_event

    def _on_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("admitted observation payload must be a dictionary")
        observation = CandidateObservation(**payload)
        self._last_material_event = self.monitor.observe(observation)


class _GenerationConnection:
    """Fence each existing journal transaction under the same SQLite write lock."""

    def __init__(self, connection, owner):
        self._connection, self._owner = connection, owner

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, sql, parameters=()):
        result = self._connection.execute(sql, parameters)
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            try:
                self._owner._check_generation()
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return result


class NautilusContinuousRuntime:
    """Isolated SANDBOX observation PoC driven by upstream live actor timers.

    Accepts already-admitted injected records, never admits provider data itself.
    The lease fences unsigned mechanics; it is not an Axiom authority grant or
    signed production fence. No client, factory, plugin, feed or transport input.
    """

    def __init__(self, journal: CandidateJournal, *, interval_ns: int = 100_000_000):
        if type(interval_ns) is not int or not 10_000_000 <= interval_ns <= 1_000_000_000:
            raise ValueError("timer interval must be 10ms through 1s")
        self.journal, self.monitor = journal, CandidateMonitor(journal)
        self._connection = journal.connection
        self.interval_ns = interval_ns
        self.lease_ns = max(500_000_000, interval_ns * 5)
        self.generation = 0
        self._started = self._closing = False
        self._node = self._actor = self._bus = self._task = None
        self._last_material_event = None
        self._connection.execute("""CREATE TABLE IF NOT EXISTS continuous_runtime (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), generation INTEGER NOT NULL,
            phase TEXT NOT NULL, heartbeat_ns INTEGER NOT NULL, lease_until_ns INTEGER NOT NULL,
            timer_callbacks INTEGER NOT NULL, data_deadline_ns INTEGER NOT NULL,
            error TEXT NOT NULL)""")

    def timestamp_ns(self):
        return self._actor.clock.timestamp_ns() if self._actor is not None else time.time_ns()

    def _row(self):
        return self._connection.execute("SELECT * FROM continuous_runtime WHERE singleton=1").fetchone()

    def _check_generation(self, generation=None):
        if generation is not None and (type(generation) is not int or generation != self.generation):
            raise RuntimeError("runtime generation fenced")
        row = self._row()
        if (not self._started or row is None or row['generation'] != self.generation
                or row['phase'] not in ('STARTING', 'RUNNING')
                or self.timestamp_ns() >= row['lease_until_ns']):
            raise RuntimeError("runtime generation inactive or fenced")

    async def start(self):
        if self.generation != 0:
            raise RuntimeError("runtime can start only once; construct a new instance")
        self.journal.replay()
        now = time.time_ns()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._row()
            if row and row['phase'] in ('STARTING', 'RUNNING') and now < row['lease_until_ns']:
                raise RuntimeError("another generation holds the runtime lease")
            self.generation = row['generation'] + 1 if row else 1
            self._connection.execute("INSERT OR REPLACE INTO continuous_runtime VALUES (1,?,?,?,?,0,0,'')",
                (self.generation, 'STARTING', now, now + 3_000_000_000))
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._started = True
        self.journal.connection = _GenerationConnection(self._connection, self)
        try:
            from nautilus_trader.common import DataActor, Environment, MessageBus
            from nautilus_trader.live import LiveNode, LiveNodeConfig
            from nautilus_trader.model import TraderId

            owner = self
            class AxiomObservationActor(DataActor):
                def on_start(actor):
                    actor.clock.set_timer_ns('axiom-observation-heartbeat', owner.interval_ns,
                                            callback=actor.on_time_event)

                def on_time_event(actor, event):
                    owner._tick()

                def on_stop(actor):
                    actor.clock.cancel_timers()

            self._ready = asyncio.get_running_loop().create_future()
            trader = TraderId('AXIOM-CONTINUOUS-POC')
            config = LiveNodeConfig(environment=Environment.SANDBOX, trader_id=trader,
                data_clients={}, exec_clients={}, plugins=[], catalogs=[],
                load_state=False, save_state=False, delay_post_stop_secs=0,
                timeout_connection_secs=0, timeout_reconciliation_secs=0,
                timeout_portfolio_secs=0, timeout_disconnection_secs=0, timeout_shutdown_secs=1)
            self._node = LiveNode.build('AXIOM-OBSERVATION-POC', config,
                                        data_factories={}, exec_factories={})
            self._actor = AxiomObservationActor()
            self._node.add_actor(self._actor)
            self._bus = MessageBus(trader, clock=self._actor.clock)
            self._bus.subscribe(NautilusReplayRuntime.TOPIC, self._on_observation)
            self._task = asyncio.create_task(self._drive())
            await asyncio.wait_for(asyncio.shield(self._ready), timeout=3)
        except BaseException:
            await self.shutdown()
            raise

    async def _drive(self):
        try:
            await self._node.run_async()
            if not self._closing:
                self._fault('native node stopped unexpectedly')
        except BaseException as exc:
            self._fault(type(exc).__name__)
            if not self._ready.done():
                self._ready.set_exception(RuntimeError('native runtime startup failed'))
            raise

    def _fault(self, reason):
        self._connection.execute("UPDATE continuous_runtime SET phase='FAULTED',error=? WHERE generation=?",
                                 (reason, self.generation))
        if self._node is not None:
            self._node.handle().stop()

    def _tick(self):
        try:
            self._check_generation()
            now = self.timestamp_ns()
            self.journal.advance_time(now)
            updated = self._connection.execute("""UPDATE continuous_runtime SET phase='RUNNING',
                heartbeat_ns=?,lease_until_ns=?,timer_callbacks=timer_callbacks+1
                WHERE generation=? AND phase IN ('STARTING','RUNNING') AND lease_until_ns>?""",
                (now, now+self.lease_ns, self.generation, now))
            if updated.rowcount != 1:
                raise RuntimeError('timer generation fenced')
            if self._node.handle().is_running and not self._ready.done():
                self._ready.set_result(None)
        except Exception as exc:
            self._fault(type(exc).__name__)
            if not self._ready.done():
                self._ready.set_exception(RuntimeError('native timer failed closed'))

    def publish(self, observation: CandidateObservation, *, generation: int):
        self._check_generation(generation)
        if not isinstance(observation, CandidateObservation):
            raise TypeError('injected input must be an admitted CandidateObservation')
        now = self.timestamp_ns()
        if observation.observed_at_ns > now:
            raise ValueError('future observation is not admitted by this runtime')
        self.journal.advance_time(now)
        self._last_material_event = None
        self._bus.publish(NautilusReplayRuntime.TOPIC, observation.to_payload(), external_pub=False)
        return self._last_material_event

    def _on_observation(self, payload):
        self._check_generation()
        observation = CandidateObservation(**payload)
        self._last_material_event = self.monitor.observe(observation)
        self._connection.execute("UPDATE continuous_runtime SET data_deadline_ns=? WHERE generation=?",
                                 (observation.freshness_deadline_ns, self.generation))

    def pending(self, *, generation: int):
        self._check_generation(generation)
        return self.journal.pending_wakeups()

    def complete(self, result, *, generation: int):
        self._check_generation(generation)
        self.journal.advance_time(self.timestamp_ns())
        return self.journal.consume_research_wakeup(result)

    def status(self):
        row = self._row()
        now = self.timestamp_ns()
        phase = row['phase'] if row else 'NOT_STARTED'
        verified = bool(self._started and row and row['generation'] == self.generation
            and phase == 'RUNNING' and now < row['lease_until_ns']
            and self._node is not None and self._node.handle().is_running)
        if phase in ('STARTING', 'RUNNING') and not verified:
            phase = 'HEARTBEAT_STALE' if row and now >= row['lease_until_ns'] else 'UNVERIFIED'
        fresh = bool(verified and now < row['data_deadline_ns'])
        ready = fresh and self._connection.execute(
            "SELECT 1 FROM candidate_states WHERE state='READY' AND freshness_deadline_ns>? LIMIT 1", (now,)).fetchone() is not None
        return dict(runtime=phase, generation=row['generation'] if row else 0,
            timer_callbacks=row['timer_callbacks'] if row else 0,
            heartbeat_ns=row['heartbeat_ns'] if row else 0, data_fresh=fresh,
            candidate_ready=bool(ready), input_scope='synthetic-or-injected-admitted-only',
            live_feed_verified=False, owner_transport_verified=False,
            execution_enabled=False, capital_authorized=False, error=row['error'] if row else '')

    async def shutdown(self):
        self._closing = True
        if self._node is not None:
            self._node.handle().stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=3)
            except Exception as exc:
                self._fault(type(exc).__name__)
                raise
        if self._actor is not None:
            self._actor.clock.cancel_timers()
        if self._bus is not None:
            self._bus.dispose()
        if self._node is not None:
            self._node.dispose()
        self._connection.execute("UPDATE continuous_runtime SET phase='STOPPED' WHERE generation=? AND phase!='FAULTED'",
                                 (self.generation,))
        self._started = False
        self.journal.connection = self._connection
        self._task = self._bus = self._actor = self._node = None
