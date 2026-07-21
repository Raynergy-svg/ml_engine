"use client";

import type { ReactNode } from "react";
import { BrainLoopPanel } from "./BrainLoopPanel";
import { ControlPanel } from "./ControlPanel";
import { CryptoCarryPanel } from "./CryptoCarryPanel";
import { CryptoMomentumPanel } from "./CryptoMomentumPanel";
import { HedgePanel } from "./HedgePanel";
import { LearningLoopPanel } from "./LearningLoopPanel";
import { RiskTrimPanel } from "./RiskTrimPanel";
import { TrackBPanel } from "./TrackBPanel";
import { Badge, Card, StatusDot } from "./ui";

const GROUPS = [
  {
    id: "operations",
    number: "01",
    label: "Live operations",
    short: "Controls",
    color: "#ffc23e",
    badge: "BACKEND READBACK",
    description:
      "Bounded controls that can change running practice processes or risk posture.",
  },
  {
    id: "learning",
    number: "02",
    label: "Autonomous learning",
    short: "Learning",
    color: "#3ee287",
    badge: "EVIDENCE + GATES",
    description:
      "Research, retraining markers, promotion requests, and de-risk decisions.",
  },
  {
    id: "shadow",
    number: "03",
    label: "Shadow strategies",
    short: "Shadow lanes",
    color: "#22d3ee",
    badge: "SHADOW ONLY",
    description:
      "Experimental crypto and Track B lanes observed without live-order authority.",
  },
  {
    id: "protection",
    number: "04",
    label: "Portfolio protection",
    short: "Protection",
    color: "#3ee287",
    badge: "PROPOSAL ONLY",
    description:
      "Concentration trimming, hedge analysis, and exposure-history coverage.",
  },
] as const;

function GroupSection({
  id,
  children,
}: {
  id: (typeof GROUPS)[number]["id"];
  children: ReactNode;
}) {
  const group = GROUPS.find((item) => item.id === id)!;
  return (
    <section id={`settings-${id}`} className="scroll-mt-40 space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b px-1 pb-2 hairline">
        <div className="flex min-w-0 items-start gap-3">
          <span className="font-mono text-[11px] text-faint">
            {group.number}
          </span>
          <div>
            <h2 className="font-mono text-[14px] font-semibold uppercase tracking-[0.13em] text-text">
              {group.label}
            </h2>
            <p className="mt-1 max-w-3xl text-[11px] leading-relaxed text-faint">
              {group.description}
            </p>
          </div>
        </div>
        <Badge color={group.color} dot>
          {group.badge}
        </Badge>
      </div>
      {children}
    </section>
  );
}

export function SettingsWorkspace() {
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[250px_minmax(0,1fr)]">
      <aside className="space-y-3 xl:sticky xl:top-40">
        <Card className="overflow-hidden">
          <div className="border-b p-4 hairline">
            <div className="eyebrow">Settings workspace</div>
            <h1 className="mt-1.5 text-[18px] font-semibold tracking-tight text-text">
              Automation & controls
            </h1>
            <p className="mt-2 text-[11px] leading-relaxed text-faint">
              Backend readback for operational controls; evidence-backed status
              for learning, shadow, and protection surfaces.
            </p>
          </div>
          <nav
            aria-label="Settings sections"
            className="grid gap-1 p-2 sm:grid-cols-2 xl:grid-cols-1"
          >
            {GROUPS.map((group) => (
              <button
                key={group.id}
                onClick={() =>
                  document
                    .getElementById(`settings-${group.id}`)
                    ?.scrollIntoView({ behavior: "smooth", block: "start" })
                }
                className="group flex items-center gap-2.5 rounded-md border border-transparent px-2.5 py-2 text-left transition-colors hover:border-border hover:bg-surface2"
              >
                <StatusDot color={group.color} />
                <span className="min-w-0 flex-1">
                  <span className="block font-mono text-[11px] text-text">
                    {group.short}
                  </span>
                  <span className="block truncate font-mono text-[9.5px] text-faint">
                    {group.badge}
                  </span>
                </span>
                <span className="font-mono text-[10px] text-faint">
                  {group.number}
                </span>
              </button>
            ))}
          </nav>
        </Card>
        <Card className="p-3">
          <div className="eyebrow mb-2">Safety legend</div>
          <div className="space-y-2 font-mono text-[10px] text-faint">
            <div>
              <b className="text-text">Write path</b>
              <br />
              requires confirmation and server-side guards
            </div>
            <div>
              <b className="text-text">Bounded</b>
              <br />
              may only maintain or reduce risk
            </div>
            <div>
              <b className="text-text">Shadow</b>
              <br />
              observes and records; never executes
            </div>
            <div>
              <b className="text-text">Unavailable</b>
              <br />
              no live state or artifact is connected
            </div>
          </div>
        </Card>
      </aside>
      <div className="min-w-0 space-y-7">
        <GroupSection id="operations">
          <ControlPanel />
        </GroupSection>
        <GroupSection id="learning">
          <BrainLoopPanel />
          <LearningLoopPanel />
        </GroupSection>
        <GroupSection id="shadow">
          <CryptoMomentumPanel />
          <TrackBPanel />
          <CryptoCarryPanel />
        </GroupSection>
        <GroupSection id="protection">
          <RiskTrimPanel />
          <HedgePanel />
        </GroupSection>
      </div>
    </div>
  );
}
