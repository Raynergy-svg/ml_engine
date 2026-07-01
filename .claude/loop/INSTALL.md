# Enforcement layer — install on a fresh clone

The deterministic enforcement (Stop-hook risk tripwire + SessionStart context boot) is wired through
`.claude/settings.json`, which is **gitignored** (it holds machine-local absolute paths and a
permissions allowlist). So the wiring does NOT travel with the repo — a fresh clone has the scripts
but not the hooks. Re-apply the wiring once per clone:

Add these to `.claude/settings.json` under `"hooks"` (merge with any existing entries). Using
`$CLAUDE_PROJECT_DIR` keeps the paths clone-location-independent:

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [
        { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/tools/session_context_boot.sh" }
      ] }
    ],
    "Stop": [
      { "hooks": [
        { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/tools/stop_gate.sh" }
      ] }
    ]
  }
}
```

Then make the scripts executable and verify:

```
chmod +x .claude/tools/stop_gate.sh .claude/tools/risk_monitor.sh .claude/tools/session_context_boot.sh \
         .claude/loop/verify_gate.py .claude/loop/loop_gate.py .claude/loop/tests/test_loop_enforcement.py
python3 .claude/loop/tests/test_loop_enforcement.py   # expect: NN passed, 0 failed
python3 .claude/loop/verify_gate.py                   # expect: VERIFY GATE: PASS
.claude/tools/risk_monitor.sh                          # expect: RISK MONITOR: GREEN, exit 0
```

Until the wiring is applied, the enforcement is ADVISORY (the scripts exist but nothing auto-runs
them). The code-level guards in `src/scanner/execution.py` (halt) and the `HARD_MAX_GAP` quarantine
remain the primary trading rail regardless — this layer is defense-in-depth.
