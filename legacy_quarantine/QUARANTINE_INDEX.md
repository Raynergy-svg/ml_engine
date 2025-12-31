# Quarantine Index

Record of legacy/duplicated/outdated items staged for quarantine.

Format
- Path: <relative path>
- Reason: <duplication/legacy/perf>
- Replacement: <new module/file>
- Owner: <who to confirm>
- Date: <YYYY-MM-DD>

Entries
- Path: README.md (sections pre-refactor, if conflicting)
  Reason: Legacy docs referencing older entrypoints
  Replacement: main.py CLI docs
  Owner: Mirela
  Date: 2025-12-29

- Path: config.yaml.bak
  Reason: Outdated backup config
  Replacement: config.yaml
  Owner: Mirela
  Date: 2025-12-29

- Path: config_tuned_combo_1.yaml .. config_tuned_combo_5.yaml
  Reason: Superseded tuning snapshots
  Replacement: config.yaml + notes in COMPLETE_IMPROVEMENTS_SUMMARY.md
  Owner: Mirela
  Date: 2025-12-29

- Path: legacy_quarantine/*
  Reason: Retired entrypoints/docs
  Replacement: main.py train-buddy/buddy
  Owner: Mirela
  Date: 2025-12-29
