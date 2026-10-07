# SDD ledger — plan: docs/superpowers/plans/2026-10-06-axiom2-nautilus-runtime.md

Base: 9e996c8bd3e0a37fafed2793be1d1165d91209b3. Isolated completion branch; remote runtime branch unchanged.

Ruling: Resume inherited Tasks 1–4 implementation instead of rewriting completed mechanics — preserve verified checkpoint and upstream pin — inherited defects require explicit regression tests.
Ruling: Keep stable handoff artifacts in docs/axiom2 instead of deleting the ledger — user explicitly requires stable handoff — additional tracked documentation.
Pre-flight: Task 2 raw observation/state/outbox consumed by Task 3; version supersession and freshness need verification.
Pre-flight: Task 4 raw events consumed by comparison; current comparison lives only in a test and needs an actual projection consumer.
Cheap baseline: smoke contract + candidate monitor, 16 passed in 0.14s.
Dependency root cause: runtime order comparison imports Axiom signed lifecycle -> pydantic and cryptography, absent from upstream make sync environment. Exact pins added only after upstream build/cache verification.
Artifact: 11454954090, run 37553124116, GitHub digest sha256:fad40bc654f5965a8bef2317036f1b9a20504138f30e2a516afcfa90414ec721. Download through storage blocked 403; connector transfer exceeds 32 MiB. Use bounded verified cloud cache; cache miss fails, no rebuild.
