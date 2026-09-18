# Documentation Archive Policy

Superseded documentation belongs here; current architecture truth does not.

Target:
```text
docs/
  architecture/
  runbooks/
  reference/
  research/
  incidents/
  archive/YYYY-MM/{architecture,implementation,handoffs,experiments,plans}/
```

Archive a document when it describes a superseded/completed phase, an obsolete handoff, or an implementation no longer active. Before moving it, search code/docs for path references and update them in the same commit.

The first cleanup pass intentionally creates taxonomy without bulk-moving 200+ files. Cosmetic reorganization must not break links while authority is still being determined.
