# Improvement Rules

Meta-rules governing how Buddy learns and evolves.

## Learning Triggers
- Every closed trade triggers learning extraction (analyze outcome vs prediction)
- Every losing trade > $100 triggers deep analysis (LLM-assisted if enabled)
- Every 10 scan cycles triggers learnings audit (consolidation check)

## Promotion Criteria
- A pattern observed 3+ times in learnings.md gets promoted to rules/trading.md
- Promoted rules include the date, source count, and specific actionable directive
- Source learnings are marked [PROMOTED] after extraction

## Consolidation
- When learnings.md exceeds 30 entries: group by category, archive old entries
- When rules/trading.md exceeds 50 lines: split by domain (entry rules vs risk rules)
- When config_adjustments.json exceeds 100 entries: archive entries older than 30 days

## Anti-Patterns
- Never create new .claude/ files without justification — edit existing ones
- Never let learnings accumulate without triage (apply / capture / dismiss)
- Never evolve config silently — log every adjustment with reason
- Never guess at stale state — read state.json, ask if unclear
