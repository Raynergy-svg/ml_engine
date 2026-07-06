# Activity Incident Queue Design

## Goal

Replace the Activity tab's redundant panels and copied settings controls with a focused incident queue. Activity should answer: what is happening now, why is it happening, what did AXIOM do automatically, and what safe operator action is available next.

## Product Shape

Activity uses a compact top status strip, then prioritized incident cards. Each card has two sides:

- Proof / Evidence: trigger, threshold, current state, AXIOM action, and a small tail of relevant log/audit lines.
- Recovery / Safe Actions: direct confirmable actions for B-level operational containment and recovery.

The tab should not show the full settings/control panel. Health, audit, decisions, and raw logs appear only as evidence inside incidents or compact secondary context.

## Action Policy

Direct Activity actions are limited to operational recovery and containment:

- acknowledge an incident
- recheck liveness/state
- pause or unpause a lane
- stop or restart a stuck whitelisted loop
- view evidence/details

Emergency trade-risk actions are AXIOM-owned, not default operator buttons in Activity:

- global halt
- flatten positions
- leverage clamp
- hard shutdown

Activity may display that AXIOM took one of these actions, why it did it, and whether it succeeded. Manual access to those controls remains in the stricter control surfaces.

## Data Flow

The frontend derives incidents from existing feeds first:

- `/api/activity` log feeds and background tasks
- `/api/system_health` alerts and gate/lane health
- `/api/tier7` heartbeat, self-heal, and snapshot state
- `/api/control/state` lane and loop status
- `/api/control/audit` recent operator and AXIOM decisions

No new trading decision path is introduced. Incident cards call existing `/api/control/{action}` endpoints for safe operational actions, still using the existing confirmation header and backend safety enforcement.

## UI Changes

Activity should remove:

- copied `ControlPanel`
- standalone Health panel
- standalone Recent Decisions card
- large always-visible raw log wall

Activity should keep:

- compact status tiles
- prioritized incident cards
- compact resolved/recent context
- raw log/evidence drawers inside incident cards

## Safety And Error Handling

All mutating actions remain confirm-gated. Actions that can change trading risk are not added as incident buttons. Denied control calls stay visible in-card with the backend denial reason. Missing feeds degrade into evidence states such as "log missing", "snapshot stale", or "heartbeat unknown" rather than fabricating liveness.

## Testing

Verify:

- TypeScript compiles.
- Focused ESLint passes for touched frontend files.
- Existing Activity feed backend tests pass.
- Browser QA confirms desktop and mobile render the incident queue without horizontal overflow.
- Activity shows incident actions without duplicating the full Automation/Settings control panel.
