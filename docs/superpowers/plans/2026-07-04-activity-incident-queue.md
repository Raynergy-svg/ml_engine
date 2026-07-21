# Activity Incident Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Activity's redundant copied control/health panels with an incident queue that pairs evidence with safe recovery actions.

**Architecture:** Keep the feature in the existing `ActivityPanel` frontend component. Derive incidents from already-polled Activity, Tier 7, health, control-state, and audit data. Use the existing `control()` client for safe operational actions only.

**Tech Stack:** Next.js React client components, TypeScript, Tailwind utility classes, existing FastAPI control endpoints.

---

### Task 1: Remove Redundant Activity Composition

**Files:**
- Modify: `dashboard/web/app/page.tsx`

- [ ] **Step 1: Remove copied control/health from Activity**

Update the Activity tab render block to:

```tsx
{tab === "Activity" && <ActivityPanel />}
```

- [ ] **Step 2: Keep full control surfaces elsewhere**

Leave Automation and Settings rendering `ControlPanel`; do not remove the stricter control surfaces.

- [ ] **Step 3: Run focused TypeScript check**

Run: `cd dashboard/web && npx tsc --noEmit`

Expected: exit code 0.

### Task 2: Add Incident Queue To ActivityPanel

**Files:**
- Modify: `dashboard/web/components/ActivityPanel.tsx`

- [ ] **Step 1: Import the existing control client**

Add:

```tsx
import { control, type ControlResult } from "@/lib/control";
```

- [ ] **Step 2: Add a local two-step confirm button**

Create a small local button that arms on first click and runs `control()` on confirm. It should display backend denial/result text inline.

- [ ] **Step 3: Derive incident cards from existing data**

Create incident candidates in priority order:

```ts
alerts -> heldFeeds -> quietActive -> tier7 heartbeat/log mismatch -> recent denied audit
```

Each incident includes severity, title, summary, evidence rows, log lines, and safe actions.

- [ ] **Step 4: Render proof/recovery split**

Each incident card renders:

```tsx
<EvidencePane rows={incident.evidence} lines={incident.lines} />
<RecoveryPane actions={incident.actions} />
```

Safe direct actions are limited to `halt`/`unhalt` with a lane param, `start_loop`, and `stop_loop`. Do not add flatten, global unhalt, leverage, or global halt as incident-card buttons.

- [ ] **Step 5: Keep raw logs secondary**

Move log feeds behind compact `<details>` drawers below incidents. Do not render the old full log grid as the primary Activity body.

### Task 3: Verification

**Files:**
- Test: `dashboard/server/test_activity_feed.py`
- Check: `dashboard/web/app/page.tsx`
- Check: `dashboard/web/components/ActivityPanel.tsx`

- [ ] **Step 1: Backend feed test**

Run: `python -m pytest dashboard/server/test_activity_feed.py`

Expected: all tests pass.

- [ ] **Step 2: Frontend checks**

Run:

```bash
cd dashboard/web
npx tsc --noEmit
npx eslint app/page.tsx components/ActivityPanel.tsx
```

Expected: exit code 0.

- [ ] **Step 3: Browser QA**

Open `http://localhost:51999`, select Activity, and verify:

- incident queue appears
- copied full `ControlPanel` is gone from Activity
- incident cards show proof and recovery side-by-side on desktop
- mobile has no horizontal overflow
- emergency trade-risk controls are not default incident-card buttons
