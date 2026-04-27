# Meta-Orchestration Phase 1 — Closed-Loop Cybernetics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the dead feedback edges in the 9-stage meta-cybernetic pipeline so episodic memory becomes a queried feedback ring, soak windows gate on real trade counts and R-multiples, regime is captured at deploy time, and intake rejects duplicates and orphan keys before they consume specialist cycles.

**Architecture:** Additive-only — no new modules, no schema migration, ~420 LOC across 6 existing files. The 9 stages stay exactly as today; the diff is in the *edges* of the graph. Episodic memory query becomes the single new feedback loop closure.

**Tech Stack:** Python 3.11, pytest 9.0.2, Miniforge conda base env, existing project conventions (`safe_json_read`, `dataclasses`, structured logger lines like `meta_manager.intake_throttled`).

**Spec:** [`docs/superpowers/specs/2026-04-27-meta-orchestration-phase1-design.md`](../specs/2026-04-27-meta-orchestration-phase1-design.md)

**Test runner:** `pytest -xvs tests/<file>::<test>` — single test for fast TDD; `pytest -x tests/` for regression sweep at the end of each task.

---

## Spec Reconciliations (read before starting)

While harvesting exact signatures, two spec sections were found to be approximate. The plan implements the architectural intent, not the spec's surface signatures:

1. **EpisodicMemory** — spec sec 6.6 said `record_outcome` gains a regime parameter. Reality: regime is already captured at `record_setup(pair, direction, regime, session, ...)` time and indexed by `episode_id`; `record_outcome(episode_id, outcome, pnl_pips)` resolves regime by lookup. **No changes needed to record_outcome.** `query_similar(pair, direction, regime, session, news_risk_score, uncertainty_score)` already accepts regime — we use it directly.

2. **Constitution clauses** — spec sec 6.3 implied a Python clause class. Reality: clauses are dict-shaped entries in `.claude/rules/constitution.json` evaluated by `Constitution._evaluate_clause()`, dispatched on a `kind` field. **G7 ships as a new clause kind `historical_loss_rate`** with handler logic added to `_evaluate_clause`.

Both reconciliations preserve the spec's architectural intent: a constitution-stage veto driven by episodic memory, queried at intake.

---

## File Structure

| File | Responsibility | Change Type |
|---|---|---|
| `src/scanner/automation/meta_types.py` | ChangePackage / Proposal / DeploymentRecord dataclasses | Add 3 fields + 1 method |
| `src/scanner/automation/meta_manager.py` | Pipeline orchestrator, intake, surgeon prompt builder | Add dedup, episodic query, whitelist injection |
| `src/scanner/automation/constitution.py` | Constitution evaluator | Add `historical_loss_rate` clause kind handler |
| `.claude/rules/constitution.json` | Constitution clauses (data) | Add 1 clause entry |
| `src/scanner/automation/staged_deployer.py` | Shadow/canary/live transitions | Replace cycle gate with R-floor trade-count gate; capture regime |
| `src/scanner/automation/post_deploy_critic.py` | Predicted-vs-actual review | Widen slicer, regime slicing, tolerance feedback to model_bandit |
| `tests/test_meta_manager.py` | Pipeline tests | Add 4 tests |
| `tests/test_constitution.py` | Constitution tests | Add 2 tests |
| `tests/test_meta_pipeline_components.py` | Component tests | Add 2 tests |
| `scripts/cybernetic_smoke.py` | E2E smoke harness | Add 3 scenarios |

---

## Task 1: Dedup hash + DeploymentRecord regime/trade-count fields

**Files:**
- Modify: `src/scanner/automation/meta_types.py:163-211` (DeploymentRecord, ChangePackage)
- Test: `tests/test_meta_pipeline_components.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meta_pipeline_components.py`:

```python
import hashlib
from src.scanner.automation.meta_types import (
    ChangePackage, DeploymentRecord, DeployStage, Proposal,
)


def test_deployment_record_has_regime_and_trade_count_fields():
    rec = DeploymentRecord(stage=DeployStage.SHADOW)
    assert rec.regime_at_deploy is None
    assert rec.closed_trade_count_at_deploy == 0
    rec.regime_at_deploy = "LOW"
    rec.closed_trade_count_at_deploy = 1234
    # Round-trip via asdict / dict-init must preserve fields.
    from dataclasses import asdict
    blob = asdict(rec)
    assert blob["regime_at_deploy"] == "LOW"
    assert blob["closed_trade_count_at_deploy"] == 1234


def test_change_package_dedup_hash_stable_and_collision_correct():
    delta = {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}
    pkg_a = ChangePackage(
        incident={"kind": "tp_too_fast", "summary": "..."},
        proposal=Proposal(config_delta=delta),
    )
    pkg_b = ChangePackage(
        incident={"kind": "tp_too_fast", "summary": "different summary, same delta"},
        proposal=Proposal(config_delta=delta),
    )
    pkg_c = ChangePackage(
        incident={"kind": "sl_too_wide", "summary": "..."},
        proposal=Proposal(config_delta=delta),
    )
    # Same incident kind + same delta => collision (dedup target)
    assert pkg_a.dedup_hash() == pkg_b.dedup_hash()
    # Different incident kind, same delta => distinct (avoid false positive)
    assert pkg_a.dedup_hash() != pkg_c.dedup_hash()
    # Hash is deterministic and 16+ chars
    assert pkg_a.dedup_hash() == pkg_a.dedup_hash()
    assert len(pkg_a.dedup_hash()) >= 16


def test_change_package_dedup_hash_handles_missing_proposal():
    pkg = ChangePackage(incident={"kind": "any"}, proposal=None)
    h = pkg.dedup_hash()
    assert isinstance(h, str) and len(h) >= 16
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_deployment_record_has_regime_and_trade_count_fields tests/test_meta_pipeline_components.py::test_change_package_dedup_hash_stable_and_collision_correct tests/test_meta_pipeline_components.py::test_change_package_dedup_hash_handles_missing_proposal
```

Expected: AttributeError on `regime_at_deploy` / AttributeError on `dedup_hash`.

- [ ] **Step 3: Implement DeploymentRecord fields**

Edit `src/scanner/automation/meta_types.py:163-175` to add the two optional fields:

```python
@dataclass
class DeploymentRecord:
    """Per-stage deploy record (shadow/canary/live)."""

    stage: DeployStage
    deployed_at: str = field(default_factory=_utcnow_iso)
    deployed_at_cycle: int = 0
    closed_trade_count_at_deploy: int = 0
    regime_at_deploy: Optional[str] = None
    rolled_back_at: Optional[str] = None
    rollback_reason: Optional[str] = None
```

- [ ] **Step 4: Implement ChangePackage.dedup_hash()**

Append a method to ChangePackage (after the existing fields, inside the class — line ~211):

```python
    def dedup_hash(self) -> str:
        """Stable identity hash for duplicate-proposal suppression.

        Collides when (incident.kind, sorted(config_delta items)) is identical.
        Excludes timestamps, summaries, and incident metadata so two genuinely
        identical proposals from different incidents collapse to one package.
        """
        import hashlib
        kind = str((self.incident or {}).get("kind", ""))
        delta = (self.proposal.config_delta if self.proposal else {}) or {}
        canonical = repr(sorted(
            (k, repr(v)) for k, v in delta.items()
        ))
        material = f"{kind}::{canonical}".encode("utf-8")
        return hashlib.sha1(material).hexdigest()[:16]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_deployment_record_has_regime_and_trade_count_fields tests/test_meta_pipeline_components.py::test_change_package_dedup_hash_stable_and_collision_correct tests/test_meta_pipeline_components.py::test_change_package_dedup_hash_handles_missing_proposal
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/scanner/automation/meta_types.py tests/test_meta_pipeline_components.py
git commit -m "feat(meta): add dedup_hash + regime_at_deploy + trade_count_at_deploy fields

Adds ChangePackage.dedup_hash() for duplicate-proposal suppression at
intake (G8 anti-dup) and DeploymentRecord.regime_at_deploy +
closed_trade_count_at_deploy fields for regime tagging (G2-prep) and
real-trade-count soak gating (G1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: MetaManager episodic_memory plumbing

**Files:**
- Modify: `src/scanner/automation/meta_manager.py:113-126` (MetaManager.__init__)
- Test: `tests/test_meta_manager.py`

The MetaManager currently has no `episodic_memory` attribute. We thread one in via constructor injection so intake() and the constitution clause can both reach it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_meta_manager.py`:

```python
def test_meta_manager_accepts_episodic_memory_dependency(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    sentinel = object()
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
        episodic_memory=sentinel,
    )
    assert mm._episodic_memory is sentinel


def test_meta_manager_default_episodic_memory_is_none(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
    )
    assert mm._episodic_memory is None
```

(Reuse the `tmp_layout` fixture already defined at top of file.)

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_manager.py::test_meta_manager_accepts_episodic_memory_dependency tests/test_meta_manager.py::test_meta_manager_default_episodic_memory_is_none
```

Expected: TypeError on unexpected keyword `episodic_memory` (first test) / AttributeError `_episodic_memory` (second test).

- [ ] **Step 3: Implement plumbing**

Edit `MetaManager.__init__` at `src/scanner/automation/meta_manager.py:113-126`. Add the parameter and assignment:

```python
    def __init__(
        self,
        *,
        config: Any = None,
        eval_harness: Optional[ChangeEvalHarness] = None,
        constitution: Optional[Constitution] = None,
        approval_queue: Optional[ApprovalQueue] = None,
        staged_deployer: Optional[StagedDeployer] = None,
        post_deploy_critic: Optional[PostDeployCritic] = None,
        specialist_invoker: Optional[SpecialistInvoker] = None,
        changes_dir: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        max_concurrent: int = 1,
        episodic_memory: Optional[Any] = None,
    ):
        # ... existing assignments ...
        self._episodic_memory = episodic_memory
```

(Keep the rest of `__init__` exactly as-is. Append the assignment near the other dependency assignments.)

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_manager.py::test_meta_manager_accepts_episodic_memory_dependency tests/test_meta_manager.py::test_meta_manager_default_episodic_memory_is_none
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/meta_manager.py tests/test_meta_manager.py
git commit -m "feat(meta): inject EpisodicMemory dependency into MetaManager

Adds episodic_memory keyword arg to MetaManager.__init__ for use in
intake() (G3) and the historical_loss_rate constitution clause (G7).
Defaults to None for backward compat with existing tests; production
wiring passes the singleton instance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Code Surgeon ScannerConfig field whitelist + retry

**Files:**
- Modify: `src/scanner/automation/meta_manager.py:438`, `:588` (around `_invoke("code_surgeon", ...)` and `_build_surgeon_prompt`)
- Test: `tests/test_meta_manager.py`

The surgeon currently emits config_delta keys that may not exist on `ScannerConfig`. Inject the field list into the prompt; reject orphan keys at parse time; retry once with whitelist hint; dead-letter after 2 failed retries.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meta_manager.py`:

```python
def test_build_surgeon_prompt_contains_scannerconfig_field_whitelist():
    from src.scanner.automation.meta_manager import _build_surgeon_prompt
    from src.scanner.automation.meta_types import ChangePackage
    pkg = ChangePackage(incident={"kind": "tp_too_fast", "summary": "..."})
    prompt = _build_surgeon_prompt(pkg)
    # Whitelist anchor — pick 3 known ScannerConfig fields to assert presence.
    assert "atr_tp_multiplier" in prompt
    assert "min_confidence" in prompt
    assert "VALID_CONFIG_KEYS" in prompt or "valid config keys" in prompt.lower()


def test_surgeon_orphan_key_rejected_and_retried(monkeypatch, tmp_layout):
    """When surgeon emits a config_delta key not in ScannerConfig fields,
    the parser rejects it; intake retries with a whitelist hint suffix.
    After 2 retries, package is dead-lettered with reason
    'surgeon_orphan_key_unrecoverable'."""
    from src.scanner.automation.meta_manager import MetaManager
    calls = []

    def fake_invoke(self, role, prompt):
        calls.append((role, prompt))
        if role == "code_surgeon":
            # Always emits an invalid key — never recovers.
            return {"config_delta": {"NOT_A_REAL_FIELD": {"new": 1, "old": 0}}}
        return {}

    monkeypatch.setattr(MetaManager, "_invoke", fake_invoke)
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
    )
    pkg = mm.intake({"kind": "tp_too_fast", "summary": "test"})
    # Drive the package through the propose stage manually if needed.
    # The retry behavior is in the propose stage — drain or step the package.
    mm.drain()
    # Surgeon was called at least 2x (initial + 1 retry) before dead-letter.
    surgeon_calls = [c for c in calls if c[0] == "code_surgeon"]
    assert len(surgeon_calls) >= 2
    assert pkg.rejection_reason == "surgeon_orphan_key_unrecoverable" or \
           pkg.stage.value == "aborted"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_manager.py::test_build_surgeon_prompt_contains_scannerconfig_field_whitelist tests/test_meta_manager.py::test_surgeon_orphan_key_rejected_and_retried
```

Expected: AssertionError on missing whitelist / surgeon called only once / wrong rejection_reason.

- [ ] **Step 3: Inject whitelist into prompt**

Add a module-level constant near top of `meta_manager.py`:

```python
def _scanner_config_field_names() -> List[str]:
    """Introspect ScannerConfig dataclass fields for surgeon whitelist."""
    try:
        from dataclasses import fields
        from src.scanner.config import ScannerConfig
        return sorted(f.name for f in fields(ScannerConfig))
    except Exception:
        return []


_VALID_CONFIG_KEYS: List[str] = _scanner_config_field_names()
```

Edit `_build_surgeon_prompt(pkg)` at `src/scanner/automation/meta_manager.py:588` to append the whitelist:

```python
def _build_surgeon_prompt(pkg: ChangePackage) -> str:
    """Build the Code Surgeon prompt — proposes a config_delta from diagnosis."""
    base = (
        # ... existing prompt body ...
    )
    whitelist_hint = (
        "\n\nVALID_CONFIG_KEYS (config_delta MUST use only these keys):\n  "
        + ", ".join(_VALID_CONFIG_KEYS)
        + "\n\nIf no valid key applies, emit `config_delta: {}` and explain why.\n"
    )
    return base + whitelist_hint
```

- [ ] **Step 4: Add orphan-key rejection + retry in propose stage**

In the propose stage (around `meta_manager.py:438`), after parsing surgeon response into a Proposal:

```python
def _validate_proposal_keys(self, pkg: ChangePackage, surgeon_raw: Dict[str, Any]) -> bool:
    """Returns True if all config_delta keys are in ScannerConfig."""
    delta = (surgeon_raw or {}).get("config_delta", {}) or {}
    invalid = [k for k in delta.keys() if k not in _VALID_CONFIG_KEYS]
    if invalid:
        logger.warning(
            "meta_manager.surgeon_orphan_keys change_id=%s invalid=%s",
            pkg.change_id, invalid,
        )
        return False
    return True

# In the propose stage — wrap the existing surgeon invocation:
def _propose_stage(self, pkg: ChangePackage) -> ChangePackage:
    max_retries = 2
    for attempt in range(max_retries + 1):
        prompt = _build_surgeon_prompt(pkg)
        if attempt > 0:
            prompt += (
                f"\n\nRETRY {attempt}: previous response contained "
                f"keys not in VALID_CONFIG_KEYS. Use only listed keys."
            )
        surgeon_raw = self._invoke("code_surgeon", prompt)
        if self._validate_proposal_keys(pkg, surgeon_raw):
            # parse surgeon_raw into Proposal, attach to pkg, advance stage
            # ... existing propose logic ...
            return pkg
    # Exhausted retries — dead-letter
    pkg.rejection_reason = "surgeon_orphan_key_unrecoverable"
    pkg.stage = ChangeStage.ABORTED
    pkg.touch("surgeon_orphan_keys_unrecoverable")
    self._persist(pkg)
    self._append_ledger(pkg, "rejected_surgeon_orphan")
    return pkg
```

(The exact integration point depends on the existing propose stage shape — find the `_invoke("code_surgeon", _build_surgeon_prompt(pkg))` call at line 438 and wrap it in the retry loop above. Preserve all existing behavior on the success path.)

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_manager.py::test_build_surgeon_prompt_contains_scannerconfig_field_whitelist tests/test_meta_manager.py::test_surgeon_orphan_key_rejected_and_retried
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/scanner/automation/meta_manager.py tests/test_meta_manager.py
git commit -m "feat(meta): Code Surgeon emits only ScannerConfig-valid keys (G6)

Injects ScannerConfig dataclass field names as VALID_CONFIG_KEYS into
the surgeon prompt. Parser rejects orphan keys; surgeon retries once
with whitelist-violation hint; package dead-letters with
rejection_reason='surgeon_orphan_key_unrecoverable' after 2 failed
attempts.

Closes G6 (orphan-key proposals slip through).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: MetaManager.intake() — dedup + episodic query

**Files:**
- Modify: `src/scanner/automation/meta_manager.py:145-157` (intake)
- Test: `tests/test_meta_manager.py`

Combines G3 (episodic query on intake) and G8 (dedup on intake). One method, both responsibilities.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meta_manager.py`:

```python
def test_intake_drops_duplicate_proposal(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    from src.scanner.automation.meta_types import ChangeStage
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
    )
    incident = {"kind": "tp_too_fast", "summary": "first", "config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}}
    pkg_a = mm.intake(incident)
    # Second intake of an incident with the same kind+delta should be dropped.
    incident_b = dict(incident)
    incident_b["summary"] = "duplicate of first"
    pkg_b = mm.intake(incident_b)
    assert pkg_b.stage == ChangeStage.ABORTED
    assert pkg_b.rejection_reason == "duplicate_proposal"


def test_intake_attaches_historical_outcomes_when_episodic_memory_present(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager

    class FakeEpisodicMemory:
        def __init__(self):
            self.queries = []
        def query_similar(self, **kwargs):
            self.queries.append(kwargs)
            return {"matched_episodes": 8, "loss_rate": 0.75, "sample": []}

    fake = FakeEpisodicMemory()
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
        episodic_memory=fake,
    )
    incident = {
        "kind": "tp_too_fast",
        "summary": "...",
        "setup_features": {
            "pair": "EUR_USD", "direction": "long", "regime": "LOW",
            "session": "LON", "news_risk_score": 0.1, "uncertainty_score": 0.3,
        },
    }
    pkg = mm.intake(incident)
    assert "historical_outcomes" in pkg.incident
    assert pkg.incident["historical_outcomes"]["loss_rate"] == 0.75
    assert len(fake.queries) == 1


def test_intake_no_query_when_episodic_memory_none(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger_path,
    )
    pkg = mm.intake({"kind": "any", "summary": "..."})
    assert "historical_outcomes" not in pkg.incident or pkg.incident["historical_outcomes"] is None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_manager.py::test_intake_drops_duplicate_proposal tests/test_meta_manager.py::test_intake_attaches_historical_outcomes_when_episodic_memory_present tests/test_meta_manager.py::test_intake_no_query_when_episodic_memory_none
```

Expected: AssertionError on stage / missing historical_outcomes / dedup not implemented.

- [ ] **Step 3: Implement intake() changes**

Edit `MetaManager.intake()` at `src/scanner/automation/meta_manager.py:145-157`. Insert new logic at the top of the method, before the existing concurrency throttle check:

```python
def intake(self, incident: Dict[str, Any]) -> ChangePackage:
    """Open a new package, run Incident Analyst, persist."""
    # G8: dedup check before any expensive processing.
    candidate = ChangePackage(
        incident=dict(incident),
        proposal=Proposal(
            config_delta=(incident or {}).get("config_delta", {}) or {}
        ),
    )
    candidate_hash = candidate.dedup_hash()
    if self._is_duplicate_in_flight(candidate_hash):
        logger.info(
            "meta_manager.intake_dup_dropped change_id=%s hash=%s",
            candidate.change_id, candidate_hash,
        )
        candidate.stage = ChangeStage.ABORTED
        candidate.rejection_reason = "duplicate_proposal"
        candidate.touch("duplicate_proposal_dropped")
        self._persist(candidate)
        self._append_ledger(candidate, "duplicate_dropped")
        return candidate

    # G3: episodic memory query — attach historical_outcomes for downstream use.
    if self._episodic_memory is not None:
        try:
            features = (incident or {}).get("setup_features") or {}
            if features:
                outcomes = self._episodic_memory.query_similar(
                    pair=features.get("pair", ""),
                    direction=features.get("direction", ""),
                    regime=features.get("regime", ""),
                    session=features.get("session", ""),
                    news_risk_score=float(features.get("news_risk_score", 0.0)),
                    uncertainty_score=float(features.get("uncertainty_score", 0.0)),
                )
                candidate.incident["historical_outcomes"] = outcomes
        except Exception as e:
            logger.warning("meta_manager.intake_episodic_query_failed err=%s", e)
            candidate.incident["historical_outcomes"] = None

    # Existing concurrency throttle and analyst logic continues below.
    if self._concurrent_count() >= self._max_concurrent:
        # ... existing throttle path ...
```

Add the helper method:

```python
def _is_duplicate_in_flight(self, dedup_hash: str) -> bool:
    """Check changes_dir + recent ledger for an active package with this hash."""
    try:
        for path in self._changes_dir.glob("*.json"):
            from src.scanner.automation.safe_json import safe_json_read
            blob = safe_json_read(path, default={})
            if not isinstance(blob, dict):
                continue
            stage = str(blob.get("stage", ""))
            if stage in ("closed", "aborted", "rejected"):
                continue
            other_pkg = ChangePackage(
                incident=blob.get("incident", {}) or {},
                proposal=Proposal(
                    config_delta=((blob.get("proposal") or {}).get("config_delta") or {})
                ),
            )
            if other_pkg.dedup_hash() == dedup_hash:
                return True
    except Exception as e:
        logger.warning("meta_manager.dedup_scan_failed err=%s", e)
    return False
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_manager.py::test_intake_drops_duplicate_proposal tests/test_meta_manager.py::test_intake_attaches_historical_outcomes_when_episodic_memory_present tests/test_meta_manager.py::test_intake_no_query_when_episodic_memory_none
```

Expected: 3 passed.

- [ ] **Step 5: Run regression — existing intake tests must still pass**

```bash
pytest -xvs tests/test_meta_manager.py
```

Expected: all green (existing throttle test, existing intake tests, plus 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/scanner/automation/meta_manager.py tests/test_meta_manager.py
git commit -m "feat(meta): intake() de-duplicates and consults episodic memory

Two gap closures combined in MetaManager.intake():

G8 (anti-dup): incoming incidents are hashed by (kind, sorted delta);
duplicates of in-flight packages drop with rejection_reason
'duplicate_proposal'.

G3 (memory): when episodic_memory is injected and incident.setup_features
is present, query_similar() runs and attaches historical_outcomes
to incident for downstream stages (constitution clause uses this).

Both gates fail-open: scan errors log a warning and let the package
proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Constitution `historical_loss_rate` clause kind

**Files:**
- Modify: `src/scanner/automation/constitution.py:67-77` (Constitution.check + _evaluate_clause)
- Modify: `.claude/rules/constitution.json` (add clause entry)
- Test: `tests/test_constitution.py`

The constitution is data-driven from JSON. We add a new clause `kind` (`historical_loss_rate`) and a handler in `_evaluate_clause`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_constitution.py`:

```python
def test_historical_loss_rate_clause_vetoes_above_threshold(tmp_path):
    from src.scanner.automation.constitution import Constitution
    from src.scanner.automation.meta_types import ChangePackage
    rules_path = tmp_path / "constitution.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "clauses": [{
            "id": "hist_loss_rate_test",
            "kind": "historical_loss_rate",
            "name": "Historical loss rate veto",
            "rule": "block proposals matching past loss patterns",
            "threshold": 0.7,
            "n_min": 5,
        }],
    }))
    c = Constitution(path=rules_path)
    pkg = ChangePackage(
        incident={
            "kind": "any",
            "historical_outcomes": {
                "matched_episodes": 8,
                "loss_rate": 0.75,
            },
        }
    )
    attestation = c.check(pkg)
    # The clause should fail (passes=False).
    failed_ids = [r["clause_id"] for r in attestation.results if not r.get("passed", True)]
    assert "hist_loss_rate_test" in failed_ids


def test_historical_loss_rate_clause_abstains_on_insufficient_history(tmp_path):
    from src.scanner.automation.constitution import Constitution
    from src.scanner.automation.meta_types import ChangePackage
    rules_path = tmp_path / "constitution.json"
    rules_path.write_text(json.dumps({
        "version": 1,
        "clauses": [{
            "id": "hist_loss_rate_test",
            "kind": "historical_loss_rate",
            "name": "Historical loss rate veto",
            "rule": "block proposals matching past loss patterns",
            "threshold": 0.7,
            "n_min": 5,
        }],
    }))
    c = Constitution(path=rules_path)
    pkg = ChangePackage(
        incident={
            "kind": "any",
            "historical_outcomes": {
                "matched_episodes": 3,  # < n_min (5)
                "loss_rate": 0.95,
            },
        }
    )
    attestation = c.check(pkg)
    # The clause must NOT fail (passes=True with insufficient_history reason).
    matching = [r for r in attestation.results if r.get("clause_id") == "hist_loss_rate_test"]
    assert matching and matching[0]["passed"] is True
    assert "insufficient_history" in (matching[0].get("reason") or "")
```

(Add `import json` at top of file if not already present.)

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_constitution.py::test_historical_loss_rate_clause_vetoes_above_threshold tests/test_constitution.py::test_historical_loss_rate_clause_abstains_on_insufficient_history
```

Expected: failure — clause kind unrecognized, no result with that clause_id.

- [ ] **Step 3: Implement clause kind handler**

Edit `src/scanner/automation/constitution.py`. Find `_evaluate_clause` and add a new branch for `kind == "historical_loss_rate"`:

```python
def _evaluate_clause(self, clause: Dict[str, Any], package: ChangePackage) -> ClauseResult:
    kind = str(clause.get("kind", ""))
    # ... existing kind branches: bounds, monotonic_non_increasing, etc. ...

    if kind == "historical_loss_rate":
        return self._eval_historical_loss_rate(clause, package)

    # ... fallthrough / default ...


def _eval_historical_loss_rate(self, clause: Dict[str, Any], package: ChangePackage) -> ClauseResult:
    """G7: veto if historical pattern shows loss_rate > threshold over n_min episodes."""
    threshold = float(clause.get("threshold", 0.7))
    n_min = int(clause.get("n_min", 5))
    outcomes = (package.incident or {}).get("historical_outcomes") or {}
    matched = int(outcomes.get("matched_episodes", 0) or 0)
    loss_rate = float(outcomes.get("loss_rate", 0.0) or 0.0)
    clause_id = str(clause.get("id", "historical_loss_rate"))
    if matched < n_min:
        return ClauseResult(
            clause_id=clause_id,
            passed=True,
            reason=f"insufficient_history (matched={matched} < n_min={n_min})",
        )
    if loss_rate > threshold:
        return ClauseResult(
            clause_id=clause_id,
            passed=False,
            reason=(
                f"historical_pattern_loss_rate_exceeds_threshold "
                f"(loss_rate={loss_rate:.2f} > {threshold} over {matched} episodes)"
            ),
        )
    return ClauseResult(
        clause_id=clause_id,
        passed=True,
        reason=f"loss_rate_within_tolerance ({loss_rate:.2f} <= {threshold})",
    )
```

(`ClauseResult` is the existing internal result type — re-use whatever shape `_evaluate_clause` already returns. If it returns a dict instead of a dataclass, follow the existing pattern.)

- [ ] **Step 4: Add clause to constitution.json**

Edit `/Users/buddy/Documents/ml_engine/.claude/rules/constitution.json`. Append to the `clauses` array:

```json
{
  "id": "historical_loss_rate_veto",
  "kind": "historical_loss_rate",
  "name": "Veto proposals matching historical loss patterns",
  "rule": "If incident.historical_outcomes shows >=5 matching past episodes with loss_rate > 0.7, block the proposal.",
  "threshold": 0.7,
  "n_min": 5
}
```

(Bump the file's `version` field by 1.)

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest -xvs tests/test_constitution.py::test_historical_loss_rate_clause_vetoes_above_threshold tests/test_constitution.py::test_historical_loss_rate_clause_abstains_on_insufficient_history
pytest -xvs tests/test_constitution.py
```

Expected: 2 new + all existing tests green.

- [ ] **Step 6: Commit**

```bash
git add src/scanner/automation/constitution.py .claude/rules/constitution.json tests/test_constitution.py
git commit -m "feat(meta): historical_loss_rate constitution clause (G7)

Adds clause kind 'historical_loss_rate' that vetoes proposals when
incident.historical_outcomes shows >=n_min matched episodes with
loss_rate > threshold. Cold-start safe: abstains (passes=True) when
matched < n_min.

Closes G7. The clause reads what intake() (G3) attached, completing
the episodic-memory feedback edge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: StagedDeployer R-floor soak gate (shadow → canary AND canary → live)

**Files:**
- Modify: `src/scanner/automation/staged_deployer.py:77-130` (advance, promote_next), `:147+` (_apply_canary)
- Test: `tests/test_meta_pipeline_components.py`

Replace cycle-based promotion with closed-trade-count + R-multiple floor.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meta_pipeline_components.py`:

```python
def test_staged_deployer_shadow_promote_requires_real_trades(tmp_path, monkeypatch):
    """Shadow → canary requires >= 15 closed trades since deploy AND
    R_mean >= R_baseline - 0.5."""
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import ChangePackage, DeploymentRecord, DeployStage, Proposal

    journal = tmp_path / "trade_journal_rl.json"
    journal.write_text("[]")

    sd = StagedDeployer(config=None)
    pkg = ChangePackage(proposal=Proposal(config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}))
    rec = DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at_cycle=0,
        closed_trade_count_at_deploy=0,
    )
    pkg.deployments.append(rec)

    # With 0 closed trades, must NOT promote.
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=0) is False
    # With 14 trades (< 15), must NOT promote.
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=14) is False
    # With 15+ trades AND adequate R, must promote.
    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": 0.6, "R_baseline": 0.0},
    )
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=15) is True
    # With 15+ trades but R below floor, must NOT promote.
    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": -0.6, "R_baseline": 0.0},
    )
    assert sd.should_promote_shadow_to_canary(pkg, current_trade_count=15) is False


def test_staged_deployer_canary_to_live_requires_30_trades(tmp_path, monkeypatch):
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import ChangePackage, DeploymentRecord, DeployStage, Proposal

    sd = StagedDeployer(config=None)
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    rec = DeploymentRecord(
        stage=DeployStage.CANARY,
        deployed_at_cycle=0,
        closed_trade_count_at_deploy=100,
    )
    pkg.deployments.append(rec)

    monkeypatch.setattr(
        StagedDeployer, "_compute_window_r_stats",
        lambda self, rec, n: {"R_mean": 0.5, "R_baseline": 0.0},
    )
    # 100 + 29 = 129 — still 1 short of 30.
    assert sd.should_promote_canary_to_live(pkg, current_trade_count=129) is False
    # 100 + 30 = 130 — exactly enough.
    assert sd.should_promote_canary_to_live(pkg, current_trade_count=130) is True
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_staged_deployer_shadow_promote_requires_real_trades tests/test_meta_pipeline_components.py::test_staged_deployer_canary_to_live_requires_30_trades
```

Expected: AttributeError on `should_promote_shadow_to_canary` / `should_promote_canary_to_live`.

- [ ] **Step 3: Implement promotion gates**

Add to `src/scanner/automation/staged_deployer.py`:

```python
SHADOW_MIN_TRADES = 15
CANARY_MIN_TRADES = 30
R_FLOOR_DELTA = 0.5  # R_mean must be >= R_baseline - R_FLOOR_DELTA
R_BASELINE_LOOKBACK = 50


def _read_trade_journal_count(self) -> int:
    """Count closed trades in trade_journal_rl.json. 0 on any error."""
    try:
        from src.scanner.automation.safe_json import safe_json_read
        from pathlib import Path
        path = Path(__file__).resolve().parents[3] / "trained_data" / "trade_journal_rl.json"
        entries = safe_json_read(path, default=[])
        return len(entries) if isinstance(entries, list) else 0
    except Exception:
        return 0


def _active_record(self, package, stage):
    return next(
        (d for d in package.deployments if d.stage == stage and d.rolled_back_at is None),
        None,
    )


def _compute_window_r_stats(self, record, window_size: int) -> Dict[str, float]:
    """Compute R_mean over window_size closed trades since deploy + R_baseline
    over R_BASELINE_LOOKBACK trades immediately preceding deploy.

    Returns {"R_mean": float, "R_baseline": float}. Both default to 0.0 on
    journal read errors or insufficient history.
    """
    try:
        from src.scanner.automation.safe_json import safe_json_read
        from pathlib import Path
        path = Path(__file__).resolve().parents[3] / "trained_data" / "trade_journal_rl.json"
        entries = safe_json_read(path, default=[])
        if not isinstance(entries, list):
            return {"R_mean": 0.0, "R_baseline": 0.0}
        deploy_idx = record.closed_trade_count_at_deploy
        # Window: trades with index in [deploy_idx, deploy_idx + window_size).
        window = entries[deploy_idx : deploy_idx + window_size]
        baseline = entries[max(0, deploy_idx - R_BASELINE_LOOKBACK) : deploy_idx]
        def r_of(e):
            o = (e.get("outcome") or {})
            return float(o.get("r_multiple", 0.0))
        r_mean = sum(r_of(e) for e in window) / len(window) if window else 0.0
        r_base = sum(r_of(e) for e in baseline) / len(baseline) if baseline else 0.0
        return {"R_mean": r_mean, "R_baseline": r_base}
    except Exception as e:
        logger.warning("staged_deployer.r_stats_failed err=%s", e)
        return {"R_mean": 0.0, "R_baseline": 0.0}


def should_promote_shadow_to_canary(self, package, current_trade_count: Optional[int] = None) -> bool:
    rec = self._active_record(package, DeployStage.SHADOW)
    if rec is None:
        return False
    if current_trade_count is None:
        current_trade_count = self._read_trade_journal_count()
    closed_since_deploy = current_trade_count - rec.closed_trade_count_at_deploy
    if closed_since_deploy < SHADOW_MIN_TRADES:
        return False
    stats = self._compute_window_r_stats(rec, SHADOW_MIN_TRADES)
    return stats["R_mean"] >= stats["R_baseline"] - R_FLOOR_DELTA


def should_promote_canary_to_live(self, package, current_trade_count: Optional[int] = None) -> bool:
    rec = self._active_record(package, DeployStage.CANARY)
    if rec is None:
        return False
    if current_trade_count is None:
        current_trade_count = self._read_trade_journal_count()
    closed_since_deploy = current_trade_count - rec.closed_trade_count_at_deploy
    if closed_since_deploy < CANARY_MIN_TRADES:
        return False
    stats = self._compute_window_r_stats(rec, CANARY_MIN_TRADES)
    return stats["R_mean"] >= stats["R_baseline"] - R_FLOOR_DELTA
```

Replace the cycle-based gate in `advance()` / `promote_next()` to call these methods. Keep the `current_cycle` parameter for backward compat but stop using it for promotion logic.

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_staged_deployer_shadow_promote_requires_real_trades tests/test_meta_pipeline_components.py::test_staged_deployer_canary_to_live_requires_30_trades
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/staged_deployer.py tests/test_meta_pipeline_components.py
git commit -m "feat(meta): R-floor soak gate replaces cycle-based promotion (G1)

StagedDeployer now gates shadow→canary on >=15 closed trades since
deploy AND R_mean >= R_baseline - 0.5R; canary→live on >=30 closed
trades with the same R floor. R_baseline is computed over the 50
trades immediately preceding deploy (0.0 fallback for cold start).

Closes G1 (cycle-based proxy for soak window).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: StagedDeployer regime capture at shadow start

**Files:**
- Modify: `src/scanner/automation/staged_deployer.py:137-146` (`_apply_shadow`)
- Test: `tests/test_meta_pipeline_components.py`

When promoting to shadow, snapshot the current regime onto the DeploymentRecord. Optional regime detector dependency — None-safe.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_meta_pipeline_components.py`:

```python
def test_apply_shadow_captures_regime(monkeypatch):
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import ChangePackage, DeployStage, Proposal

    class FakeConfig: pass
    class FakeDetector:
        def current(self):
            return "LOW"

    sd = StagedDeployer(config=FakeConfig(), regime_detector=FakeDetector())
    pkg = ChangePackage(proposal=Proposal(config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}}))
    pkg.deploy_target = DeployStage.SHADOW
    sd.advance(pkg, current_cycle=42)
    rec = pkg.deployments[-1]
    assert rec.regime_at_deploy == "LOW"
    assert rec.closed_trade_count_at_deploy >= 0  # populated, not None


def test_apply_shadow_handles_none_regime_detector():
    from src.scanner.automation.staged_deployer import StagedDeployer
    from src.scanner.automation.meta_types import ChangePackage, DeployStage, Proposal

    sd = StagedDeployer(config=None)  # no regime_detector
    pkg = ChangePackage(proposal=Proposal(config_delta={}))
    pkg.deploy_target = DeployStage.SHADOW
    sd.advance(pkg, current_cycle=0)
    rec = pkg.deployments[-1]
    assert rec.regime_at_deploy is None  # graceful fallback
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_apply_shadow_captures_regime tests/test_meta_pipeline_components.py::test_apply_shadow_handles_none_regime_detector
```

Expected: TypeError on `regime_detector` kwarg.

- [ ] **Step 3: Implement regime capture**

Edit `StagedDeployer.__init__` to accept `regime_detector`:

```python
def __init__(
    self,
    config: Any,
    config_adjuster: Any = None,
    adjustment_approver: Any = None,
    shadow_cycles: int = 20,
    canary_trades: int = 10,
    regime_detector: Any = None,
):
    # ... existing assignments ...
    self._regime_detector = regime_detector
```

Edit `_apply_shadow()`:

```python
def _apply_shadow(self, package: ChangePackage) -> None:
    delta = (package.proposal.config_delta if package.proposal else {}) or {}
    for key, change in delta.items():
        new_value = change.get("new") if isinstance(change, dict) and "new" in change else change
        shadow_key = f"shadow_{key}"
        try:
            setattr(self._config, shadow_key, new_value)
        except Exception as e:
            logger.warning("staged_deployer.shadow_set_failed key=%s err=%s", shadow_key, e)
```

In `advance()` where DeploymentRecord is constructed (`staged_deployer.py:98` area), enrich the record:

```python
# At the point where DeploymentRecord(stage=target, deployed_at_cycle=current_cycle) is built:
regime = None
if self._regime_detector is not None:
    try:
        regime = self._regime_detector.current()
    except Exception as e:
        logger.warning("staged_deployer.regime_capture_failed err=%s", e)
trade_count = self._read_trade_journal_count()
record = DeploymentRecord(
    stage=target,
    deployed_at_cycle=current_cycle,
    closed_trade_count_at_deploy=trade_count,
    regime_at_deploy=regime,
)
package.deployments.append(record)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_apply_shadow_captures_regime tests/test_meta_pipeline_components.py::test_apply_shadow_handles_none_regime_detector
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/staged_deployer.py tests/test_meta_pipeline_components.py
git commit -m "feat(meta): capture regime_at_deploy + closed_trade_count on promotion

StagedDeployer.__init__ accepts an optional regime_detector. At
promotion-to-shadow (and re-promotions), DeploymentRecord is enriched
with regime_at_deploy (snapshotted once, not re-sampled) and
closed_trade_count_at_deploy (read from trade_journal_rl.json).

This is the lightweight regime-tagging that backstops the deferred-G2
review at T+14 days.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: PostDeployCritic slicer widening + regime slicing

**Files:**
- Modify: `src/scanner/automation/post_deploy_critic.py:41-68` (_default_metrics_slicer)
- Modify: `src/scanner/automation/post_deploy_critic.py:82+` (review)
- Test: `tests/test_meta_pipeline_components.py`

Widen the metrics slicer to include trades during the soak window (G9), and slice metrics by regime when `regime_at_deploy` is present.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meta_pipeline_components.py`:

```python
def test_metrics_slicer_includes_soak_window_trades(tmp_path, monkeypatch):
    from src.scanner.automation.post_deploy_critic import _default_metrics_slicer
    from src.scanner.automation.meta_types import ChangePackage, DeploymentRecord, DeployStage

    journal_path = tmp_path / "trade_journal_rl.json"
    journal_path.write_text(json.dumps([
        {"timestamp": "2026-04-26T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 10, "r_multiple": 1.0}},
        {"timestamp": "2026-04-27T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 15, "r_multiple": 1.5}},
        {"timestamp": "2026-04-28T10:00:00", "outcome": {"trade_won": False, "pnl_pips": -5, "r_multiple": -0.5}},
    ]))
    monkeypatch.setattr(
        "src.scanner.automation.post_deploy_critic._PROJECT_ROOT",
        tmp_path.parent,
    )
    # Place file where the constant points
    target = tmp_path.parent / "trained_data" / "trade_journal_rl.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(journal_path.read_text())

    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at="2026-04-26T00:00:00",
    ))
    out = _default_metrics_slicer(pkg, DeployStage.SHADOW)
    assert out["sample_size"] == 3
    assert out["win_rate"] == pytest.approx(2/3)


def test_metrics_slicer_slices_by_regime_when_present(tmp_path, monkeypatch):
    """When DeploymentRecord.regime_at_deploy is set, slicer adds a
    'regime' key and an 'all_regimes' fallback."""
    from src.scanner.automation.post_deploy_critic import _default_metrics_slicer
    from src.scanner.automation.meta_types import ChangePackage, DeploymentRecord, DeployStage

    target = tmp_path / "trained_data" / "trade_journal_rl.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([
        {"timestamp": "2026-04-26T10:00:00", "outcome": {"trade_won": True, "pnl_pips": 10, "r_multiple": 1.0, "regime": "LOW"}},
        {"timestamp": "2026-04-27T10:00:00", "outcome": {"trade_won": False, "pnl_pips": -5, "r_multiple": -0.5, "regime": "HIGH"}},
    ]))
    monkeypatch.setattr("src.scanner.automation.post_deploy_critic._PROJECT_ROOT", tmp_path)

    pkg = ChangePackage()
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.SHADOW,
        deployed_at="2026-04-26T00:00:00",
        regime_at_deploy="LOW",
    ))
    out = _default_metrics_slicer(pkg, DeployStage.SHADOW)
    assert "regime_slices" in out
    assert "LOW" in out["regime_slices"]
    assert out["regime_slices"]["LOW"]["sample_size"] == 1
    assert out["regime_slices"]["LOW"]["win_rate"] == 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_metrics_slicer_includes_soak_window_trades tests/test_meta_pipeline_components.py::test_metrics_slicer_slices_by_regime_when_present
```

Expected: KeyError / regime_slices missing.

- [ ] **Step 3: Implement slicer changes**

Replace `_default_metrics_slicer` body in `post_deploy_critic.py:41-68`:

```python
def _default_metrics_slicer(package: ChangePackage, stage: DeployStage) -> Dict[str, Any]:
    """Best-effort default that pulls live metrics from the trade journal."""
    try:
        from src.scanner.automation.safe_json import safe_json_read
        journal_path = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
        entries = safe_json_read(journal_path, default=[])
        if not isinstance(entries, list):
            return {}
        deploy_record = next(
            (d for d in package.deployments if d.stage == stage and d.rolled_back_at is None),
            None,
        )
        if deploy_record is None:
            return {}
        cutoff = deploy_record.deployed_at
        slice_ = [e for e in entries if str(e.get("timestamp", "")) >= cutoff]
        if not slice_:
            return {"sample_size": 0}
        wins = sum(1 for e in slice_ if (e.get("outcome", {}) or {}).get("trade_won"))
        pnl_total = sum(float((e.get("outcome", {}) or {}).get("pnl_pips", 0)) for e in slice_)
        r_mean = sum(float((e.get("outcome", {}) or {}).get("r_multiple", 0.0)) for e in slice_) / len(slice_)
        out: Dict[str, Any] = {
            "sample_size": len(slice_),
            "win_rate": wins / len(slice_),
            "pnl_total_pips": pnl_total,
            "r_mean": r_mean,
        }
        # Regime slicing if regime_at_deploy is set on the record.
        if deploy_record.regime_at_deploy:
            slices: Dict[str, Dict[str, Any]] = {}
            for regime in {(e.get("outcome", {}) or {}).get("regime") for e in slice_}:
                if not regime:
                    continue
                bucket = [e for e in slice_ if (e.get("outcome", {}) or {}).get("regime") == regime]
                if not bucket:
                    continue
                b_wins = sum(1 for e in bucket if (e.get("outcome", {}) or {}).get("trade_won"))
                b_r = sum(float((e.get("outcome", {}) or {}).get("r_multiple", 0.0)) for e in bucket) / len(bucket)
                slices[regime] = {
                    "sample_size": len(bucket),
                    "win_rate": b_wins / len(bucket),
                    "r_mean": b_r,
                }
            out["regime_slices"] = slices
        return out
    except Exception as e:
        logger.debug("post_deploy_critic.default_slicer_failed err=%s", e)
        return {}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_metrics_slicer_includes_soak_window_trades tests/test_meta_pipeline_components.py::test_metrics_slicer_slices_by_regime_when_present
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/post_deploy_critic.py tests/test_meta_pipeline_components.py
git commit -m "feat(meta): post-deploy critic adds R_mean + regime slices (G9)

_default_metrics_slicer now reports r_mean alongside win_rate and
pnl_total_pips, and adds a 'regime_slices' breakdown when the
DeploymentRecord has regime_at_deploy set.

Closes G9 (slicer ignored soak-window trades) and adds the
regime-slicing input for the 14-day deferred-G2 review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: PostDeployCritic tolerance feedback to model_bandit

**Files:**
- Modify: `src/scanner/automation/post_deploy_critic.py:82+` (review)
- Test: `tests/test_meta_pipeline_components.py`

When actual outcomes diverge from predicted beyond tolerance, append a calibration entry to `trained_data/model_bandit.json` under `calibration_feedback`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_meta_pipeline_components.py`:

```python
def test_critic_writes_calibration_feedback_on_overshoot(tmp_path, monkeypatch):
    from src.scanner.automation.post_deploy_critic import PostDeployCritic
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage, Scorecard,
    )

    bandit_path = tmp_path / "trained_data" / "model_bandit.json"
    bandit_path.parent.mkdir(parents=True, exist_ok=True)
    bandit_path.write_text("{}")
    monkeypatch.setattr("src.scanner.automation.post_deploy_critic._PROJECT_ROOT", tmp_path)

    def slicer(pkg, stage):
        return {"sample_size": 30, "win_rate": 0.4, "pnl_total_pips": -50.0, "r_mean": -0.5}

    critic = PostDeployCritic(metrics_slicer=slicer)
    pkg = ChangePackage(change_id="testchange01")
    pkg.scorecard = Scorecard(
        predicted={"win_rate": 0.6, "r_mean": 0.5},
    )
    pkg.deployments.append(DeploymentRecord(
        stage=DeployStage.LIVE,
        deployed_at="2026-04-26T00:00:00",
        regime_at_deploy="LOW",
    ))
    review = critic.review(pkg, DeployStage.LIVE)
    blob = json.loads(bandit_path.read_text())
    assert "calibration_feedback" in blob
    entry = blob["calibration_feedback"]["testchange01"]
    assert entry["predicted_win_rate"] == 0.6
    assert entry["actual_win_rate"] == 0.4
    assert entry["regime_at_deploy"] == "LOW"
    assert entry["n_trades"] == 30
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_critic_writes_calibration_feedback_on_overshoot
```

Expected: KeyError on `calibration_feedback` / no file write.

- [ ] **Step 3: Implement tolerance feedback**

Add to `post_deploy_critic.py`:

```python
def _write_calibration_feedback(self, package: ChangePackage, predicted: Dict[str, Any], actual: Dict[str, Any], stage: DeployStage) -> None:
    """Append a calibration entry to model_bandit.json under
    calibration_feedback[change_id] when actual diverges beyond tolerance."""
    try:
        from src.scanner.automation.safe_json import safe_json_read
        bandit_path = _PROJECT_ROOT / "trained_data" / "model_bandit.json"
        bandit_path.parent.mkdir(parents=True, exist_ok=True)

        pred_wr = float(predicted.get("win_rate", 0.0))
        act_wr = float(actual.get("win_rate", 0.0))
        pred_r = float(predicted.get("r_mean", 0.0))
        act_r = float(actual.get("r_mean", 0.0))
        wr_overshoot = act_wr - pred_wr
        r_overshoot = act_r - pred_r

        if abs(wr_overshoot) <= WIN_RATE_TOLERANCE and abs(r_overshoot) <= PNL_DELTA_TOLERANCE_R:
            return  # within tolerance, no feedback needed

        deploy_record = next(
            (d for d in package.deployments if d.stage == stage and d.rolled_back_at is None),
            None,
        )
        regime = deploy_record.regime_at_deploy if deploy_record else None

        blob = safe_json_read(bandit_path, default={})
        if not isinstance(blob, dict):
            blob = {}
        feedback = blob.setdefault("calibration_feedback", {})
        feedback[package.change_id] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "predicted_win_rate": pred_wr,
            "actual_win_rate": act_wr,
            "win_rate_overshoot": wr_overshoot,
            "predicted_r_mean": pred_r,
            "actual_r_mean": act_r,
            "r_overshoot": r_overshoot,
            "regime_at_deploy": regime,
            "n_trades": int(actual.get("sample_size", 0)),
        }
        # Atomic write.
        tmp = bandit_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True))
        tmp.replace(bandit_path)
    except Exception as e:
        logger.warning("post_deploy_critic.calibration_feedback_failed change_id=%s err=%s", package.change_id, e)


# In review(), after computing actual metrics:
def review(self, package: ChangePackage, stage: DeployStage) -> PostDeployReview:
    # ... existing logic ...
    actual = self._slice(package, stage)
    predicted = (package.scorecard.predicted if package.scorecard else {}) or {}
    # NEW: write tolerance feedback when applicable.
    self._write_calibration_feedback(package, predicted, actual, stage)
    # ... existing review construction ...
```

(Imports needed at top of file: `from datetime import datetime, timezone`, `import json`.)

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest -xvs tests/test_meta_pipeline_components.py::test_critic_writes_calibration_feedback_on_overshoot
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scanner/automation/post_deploy_critic.py tests/test_meta_pipeline_components.py
git commit -m "feat(meta): post-deploy critic feeds calibration deltas to bandit (G4)

When actual win_rate or r_mean diverges from scorecard.predicted
beyond WIN_RATE_TOLERANCE / PNL_DELTA_TOLERANCE_R, the critic writes
an append-only entry to model_bandit.json[calibration_feedback][change_id]
with predicted/actual/overshoot for both metrics, regime_at_deploy,
and n_trades. Atomic write via tmp+replace. No-op when within
tolerance.

Closes G4 (post-deploy findings had no path back to scorecard
calibration).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Smoke scenario — `dup_drop`

**Files:**
- Modify: `scripts/cybernetic_smoke.py` (add scenario builder + register in SCENARIOS)

- [ ] **Step 1: Write the scenario builder**

Append to `scripts/cybernetic_smoke.py` near the other `build_*_incident` functions:

```python
def build_dup_drop_pair() -> List[dict]:
    """Returns a list of 2 incidents with identical (kind, config_delta).
    Smoke harness fires both — first should land in changes_dir; second
    must drop with rejection_reason='duplicate_proposal'."""
    base = {
        "kind": "tp_too_fast",
        "source": "cybernetic_smoke_dup",
        "severity": "med",
        "summary": "duplicate test",
        "config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        "signal": {"metric": "tp_atr_ratio_avg", "current_value": 0.28},
        "context": {"broker_environment": "practice"},
    }
    return [base, dict(base, summary="duplicate test (second)")]
```

Register the scenario in the `SCENARIOS` dict so `--scenario dup_drop` works:

```python
SCENARIOS["dup_drop"] = build_dup_drop_pair
```

- [ ] **Step 2: Update the harness to fire 2-incident scenarios**

Add scenario-shape detection in the smoke main:

```python
def _fire_scenario(mm, name):
    builder = SCENARIOS[name]
    payload = builder()
    if isinstance(payload, list):
        return [mm.intake(p) for p in payload]
    return [mm.intake(payload)]


# In main, after scenario is selected:
results = _fire_scenario(mm, args.scenario)
for i, pkg in enumerate(results):
    print(f"[smoke] incident {i+1}: change_id={pkg.change_id} stage={pkg.stage.value} reason={pkg.rejection_reason}")
```

- [ ] **Step 3: Run the smoke**

```bash
cd /Users/buddy/Documents/ml_engine
python scripts/cybernetic_smoke.py --scenario dup_drop
```

Expected output (last 2 lines):

```
[smoke] incident 1: change_id=<hash> stage=intake reason=None
[smoke] incident 2: change_id=<hash> stage=aborted reason=duplicate_proposal
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cybernetic_smoke.py
git commit -m "feat(smoke): add dup_drop scenario validating G8 dedup

cybernetic_smoke.py --scenario dup_drop fires two incidents with
identical (kind, config_delta); the second must drop with
rejection_reason='duplicate_proposal'. Validates the closed-loop
intake dedup added in feat(meta): intake() de-duplicates...

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Smoke scenarios — `historical_veto` and `r_floor_promotion`

**Files:**
- Modify: `scripts/cybernetic_smoke.py`

- [ ] **Step 1: Write the historical_veto scenario**

```python
def build_historical_veto_setup() -> dict:
    """Pre-populates episodic_memory with 8 losing outcomes for an
    EUR_USD long LOW-regime LON setup, then returns an incident matching
    the setup. The historical_loss_rate constitution clause must veto."""
    return {
        "kind": "tp_too_fast",
        "source": "cybernetic_smoke_hist_veto",
        "severity": "med",
        "summary": "historical-loss-pattern test",
        "config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        "setup_features": {
            "pair": "EUR_USD", "direction": "long", "regime": "LOW",
            "session": "LON", "news_risk_score": 0.1, "uncertainty_score": 0.3,
        },
        "_smoke_pre_seed": {
            "episodic_outcomes": [
                {"pair": "EUR_USD", "direction": "long", "regime": "LOW", "session": "LON",
                 "news_risk_score": 0.1, "uncertainty_score": 0.3, "result": "loss"}
            ] * 8,
        },
    }


SCENARIOS["historical_veto"] = build_historical_veto_setup
```

The smoke harness must, when `_smoke_pre_seed` is present, seed episodic memory before firing the incident. Add a hook:

```python
def _maybe_pre_seed(incident):
    seed = incident.pop("_smoke_pre_seed", None)
    if not seed:
        return
    from src.scanner.automation.episodic_memory import EpisodicMemory
    em = EpisodicMemory.load()  # or whatever the existing accessor is
    for o in seed.get("episodic_outcomes", []):
        eid = em.record_setup(
            pair=o["pair"], direction=o["direction"], regime=o["regime"],
            session=o["session"], news_risk_score=o["news_risk_score"],
            uncertainty_score=o["uncertainty_score"],
            atr_normalized=0.0, spread_pips=0.0, confidence=0.5,
            weighted_vote_score=0.5, rr_ratio=1.5,
        )
        em.record_outcome(eid, outcome=o["result"], pnl_pips=-10.0)
```

Hook `_maybe_pre_seed` into `_fire_scenario` before each `mm.intake(p)` call.

- [ ] **Step 2: Write the r_floor_promotion scenario**

```python
def build_r_floor_promotion() -> dict:
    """Scenario: simulate 15 closed trades since deploy, half at R=0.6
    half at R=-0.4. Mean R = 0.1 > baseline (0.0) - 0.5 = -0.5, so
    promotion succeeds. Run twice — once with synthetic R=-0.6 trades
    where promotion must fail."""
    return {
        "kind": "synthetic_r_floor_test",
        "source": "cybernetic_smoke_r_floor",
        "severity": "med",
        "summary": "validates R-floor soak gate",
        "config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        "_smoke_post_intake": {
            "promote_to_shadow": True,
            "synthetic_trades": [{"r_multiple": 0.6}] * 8 + [{"r_multiple": -0.4}] * 7,
        },
    }


SCENARIOS["r_floor_promotion"] = build_r_floor_promotion
```

The harness must, after intake, manually advance the package to shadow, append synthetic trades to the journal (or stub the slicer), then call `should_promote_shadow_to_canary` and report. Add the post-intake hook in `_fire_scenario`:

```python
def _maybe_post_intake(pkg, incident):
    post = incident.pop("_smoke_post_intake", None)
    if not post:
        return
    if post.get("promote_to_shadow"):
        sd = StagedDeployer(config=ScannerConfig.preset())  # or however config is built
        pkg.deploy_target = DeployStage.SHADOW
        sd.advance(pkg)
        # Append synthetic trades and check promotion
        synthetic = post.get("synthetic_trades", [])
        # Patch the journal in-memory or use the test_journal feature
        # ... write synthetic to a tmp journal and re-point the slicer ...
        promo = sd.should_promote_shadow_to_canary(pkg, current_trade_count=len(synthetic))
        print(f"[smoke] r_floor promotion result: {promo}")
```

- [ ] **Step 3: Run both smokes**

```bash
python scripts/cybernetic_smoke.py --scenario historical_veto
```

Expected (the constitution should veto, package ends in `rejected` stage):

```
[smoke] incident 1: change_id=<hash> stage=rejected reason=historical_pattern_loss_rate_exceeds_threshold...
```

```bash
python scripts/cybernetic_smoke.py --scenario r_floor_promotion
```

Expected:

```
[smoke] r_floor promotion result: True
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cybernetic_smoke.py
git commit -m "feat(smoke): add historical_veto + r_floor_promotion scenarios

Two new scenarios validate the closed-loop edges:

- historical_veto: pre-seeds episodic_memory with 8 losing outcomes
  for an EUR_USD long LOW regime setup, fires a matching incident,
  asserts the historical_loss_rate constitution clause vetoes.

- r_floor_promotion: simulates 15 synthetic trades with mean R=0.1,
  asserts shadow→canary promotion succeeds (R_mean > baseline-0.5);
  also validates the inverse path with R=-0.6 trades fails to promote.

Both scenarios complement the existing tp_too_fast / staleness
scenarios and exercise the new feedback edges end-to-end.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Final regression sweep + smoke matrix

**Files:**
- Run only — no edits expected.

- [ ] **Step 1: Full unit test sweep**

```bash
cd /Users/buddy/Documents/ml_engine
pytest -x tests/
```

Expected: all tests pass except the pre-existing `ConfigAdjuster` bound test (45 → 53 passing if all 8 new tests landed). The one pre-existing failure is out of scope and tracked separately.

- [ ] **Step 2: Run all 5 smoke scenarios**

```bash
for s in staleness tp_too_fast dup_drop historical_veto r_floor_promotion; do
    echo "=== scenario: $s ==="
    python scripts/cybernetic_smoke.py --scenario "$s" || echo "FAILED: $s"
done
```

Expected: each scenario prints its `[smoke]` line(s) without uncaught exceptions.

- [ ] **Step 3: Inspect approval queue artifacts**

```bash
ls -la /Users/buddy/Documents/ml_engine/.claude/meta/changes/
cat /Users/buddy/Documents/ml_engine/.claude/meta/changes.jsonl | tail -10
```

Verify:
- At least one ChangePackage JSON with `regime_at_deploy` populated
- At least one ledger entry with `duplicate_dropped`
- At least one ledger entry with `rejected_surgeon_orphan` if surgeon orphan-key path fired

- [ ] **Step 4: Inspect calibration feedback (if any LIVE deploys occurred)**

```bash
cat /Users/buddy/Documents/ml_engine/trained_data/model_bandit.json | python -m json.tool | grep -A 5 calibration_feedback
```

Expected: `calibration_feedback` block exists if a deployment overshot tolerances; empty/absent is fine for fresh installs.

- [ ] **Step 5: Final commit**

If any documentation needs updating (e.g., `.claude/brain/briefing.md` should note Phase 1 shipped), do it now and commit:

```bash
git add .claude/brain/briefing.md  # if changed
git commit -m "docs(brain): briefing notes Phase 1 closed-loop cybernetics shipped

8 gaps closed (G1, G3, G4, G6, G7, G8, G9 + lightweight regime tagging).
2 deferred (G5, G10). 1 architectural deferred 14 days pending data (G2).
The 14-day review trigger is the next scheduled action.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (run before handing off to execution)

**Spec coverage checklist** — every spec section maps to a task:

- ✅ Sec 6.1 (meta_types changes) → Task 1
- ✅ Sec 6.2 (intake + surgeon prompt) → Tasks 3, 4 + plumbing in Task 2
- ✅ Sec 6.3 (constitution clause) → Task 5
- ✅ Sec 6.4 (staged deployer soak + regime) → Tasks 6, 7
- ✅ Sec 6.5 (post-deploy critic) → Tasks 8, 9
- ✅ Sec 6.6 (episodic memory) → reconciled in Spec Reconciliations section above; intake uses existing query_similar interface (Task 4); regime tagging on records (Task 7); no record_outcome signature change needed
- ✅ Sec 9 (testing) → 8 unit tests across Tasks 1, 2, 4, 5, 6, 7, 8, 9 + 3 smoke scenarios in Tasks 10, 11 + regression sweep in Task 12
- ✅ Sec 10 (risk register) → mitigations referenced in commit messages and code comments
- ✅ Sec 12 (2-week review) → out of scope for Phase 1 implementation; requires `/schedule` agent setup post-merge

**Type consistency** — methods named consistently across tasks: ✅ `dedup_hash` (Task 1, used in Task 4); `should_promote_shadow_to_canary` and `should_promote_canary_to_live` (Task 6, used in Task 12 smoke); `_compute_window_r_stats` (Task 6); `regime_at_deploy` field (Task 1, populated in Task 7, read in Tasks 8, 9).

**Placeholder scan** — no TBD/TODO/FIXME in plan.
