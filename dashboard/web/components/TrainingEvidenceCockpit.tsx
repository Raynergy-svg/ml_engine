"use client";

import { useMemo, useState } from "react";
import { usePoll } from "@/lib/api";
import { evidenceTrainingAction } from "@/lib/control";
import { ago, shortTime } from "@/lib/format";
import type {
  AxiomTrainingCockpit,
  EvidenceLaneView,
  ForwardLaneMonitor,
  TrainingDataTier,
} from "@/lib/types";
import { Badge, Card, Loading, NotConnected, SectionTitle, StatusDot } from "./ui";

const POS = "#2bd17e";
const NEG = "#ff5c7a";
const WARN = "#f5b14c";
const MUTE = "#5a6677";

function shortDigest(value?: string | null): string {
  return value ? `${value.slice(0, 12)}…` : "—";
}

function number(value?: number | null, digits = 2): string {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

function stateColor(state?: string | null): string {
  if (state === "CHAMPION" || state === "OPERATOR_APPROVED" || state === "QUARANTINED" || state === "completed" || state === "accruing" || state === "running") return POS;
  if (state === "REJECTED" || state === "RETIRED" || state === "failed" || state === "stalled" || state === "dead" || state === "health_stale") return NEG;
  if (state === "SHADOW" || state === "submitted" || state === "delayed") return WARN;
  return MUTE;
}

function parseObject(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    return parsed != null && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function tier(domain: { tiers: TrainingDataTier[] }, name: string): TrainingDataTier | undefined {
  return domain.tiers.find((item) => item.tier === name);
}

function DataPanel({ data }: { data: AxiomTrainingCockpit["data"] }) {
  const p2 = data.p2_readiness;
  const capture = data.capture;
  const tick = capture.tick;
  const exposureCapture = capture.exposure;
  const ticks = Object.values(p2.tick_trading_days_by_pair ?? {});
  const minTick = ticks.length ? Math.min(...ticks) : 0;
  const readyTickPairs = ticks.filter((days) => days >= (p2.minimum_trading_days ?? 60)).length;
  const target = p2.minimum_trading_days ?? 60;
  const exposure = p2.exposure_trading_days ?? 0;
  const progress = Math.min(100, (Math.min(exposure, minTick) / Math.max(target, 1)) * 100);
  const captureAccruing = tick.status === "accruing" && exposureCapture.status === "accruing";
  const activeDomains = data.domains.filter((domain) => domain.available || domain.tiers.some((item) => item.manifest_count));
  return (
    <Card>
      <SectionTitle right={
        <span className="flex flex-wrap gap-2">
          <Badge color={captureAccruing ? POS : stateColor(tick.status)} dot>{captureAccruing ? "CAPTURE ACCRUING" : "CAPTURE DEGRADED"}</Badge>
          <Badge color={data.quality_failure_count ? NEG : POS} dot>{data.quality_failure_count} QUALITY FAILURE{data.quality_failure_count === 1 ? "" : "S"}</Badge>
          <Badge color={p2.ready ? POS : WARN} dot>{p2.ready ? "P2 READY" : "P2 BLOCKED"}</Badge>
        </span>
      }>Data · live accrual, quality, and eligibility</SectionTitle>
      <div className="grid gap-3 p-3 lg:grid-cols-3">
        <div className="rounded-md border p-3 hairline">
          <div className="mb-2 flex items-center justify-between gap-2"><div className="eyebrow">Live tick and spread accrual</div><Badge color={stateColor(tick.status)} dot>{tick.status.toUpperCase()}</Badge></div>
          <div className="space-y-1.5 font-mono text-[10.5px]">
            <div className="flex justify-between gap-3"><span className="text-faint">required pairs current</span><span>{tick.pairs_current} / {tick.required_pairs}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">canonical batches</span><span>{tick.canonical_batches.toLocaleString()}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">last canonical commit</span><span>{shortTime(tick.last_canonical_at)}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">writer heartbeat</span><span className={tick.writer_status === "running" ? "text-pos" : "text-warn"}>{tick.writer_status} · {shortTime(tick.health_updated_at)}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">daemon session</span><span>{tick.ticks_received_session?.toLocaleString() ?? "—"} received · {tick.ticks_per_minute ?? "—"}/min</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">buffered</span><span>{tick.buffered_ticks?.toLocaleString() ?? "—"}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">flush errors this session</span><span className={tick.flush_errors ? "text-neg" : "text-pos"}>{tick.flush_errors ?? 0}</span></div>
          </div>
          {tick.last_flush_error && <div className="mt-2 rounded border border-neg/30 bg-neg/5 p-2 font-mono text-[10px] text-neg">{tick.last_flush_error.type}: {tick.last_flush_error.message}</div>}
        </div>
        <div className="rounded-md border p-3 hairline">
          <div className="mb-2 flex items-center justify-between gap-2"><div className="eyebrow">Portfolio exposure accrual</div><Badge color={stateColor(exposureCapture.status)} dot>{exposureCapture.status.toUpperCase()}</Badge></div>
          <div className="space-y-1.5 font-mono text-[10.5px]">
            <div className="flex justify-between gap-3"><span className="text-faint">canonical snapshots</span><span>{exposureCapture.canonical_snapshots.toLocaleString()}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">last canonical snapshot</span><span>{shortTime(exposureCapture.last_canonical_at)}</span></div>
            <div className="flex justify-between gap-3"><span className="text-faint">writer cadence</span><span>checks 15m</span></div>
          </div>
          <div className="mt-3 font-mono text-[10px] leading-relaxed text-faint">A new immutable row is appended only when the upstream account snapshot changes, normally about hourly. Expected dedupe between checks is not a stalled writer.</div>
        </div>
        <div className="rounded-md border p-3 hairline">
          <div className="mb-2 flex items-center justify-between gap-2"><div className="eyebrow">Risk-target P2 duration gate</div><Badge color={p2.ready ? POS : WARN}>{p2.ready ? "READY" : "BLOCKED"}</Badge></div>
          <div className="space-y-1.5 font-mono text-[10.5px]">
            <div className="flex justify-between"><span className="text-faint">exposure weekdays</span><span>{exposure} / {target}</span></div>
            <div className="flex justify-between"><span className="text-faint">minimum tick weekdays</span><span>{minTick} / {target}</span></div>
            <div className="flex justify-between"><span className="text-faint">pairs at minimum</span><span>{readyTickPairs} / {ticks.length}</span></div>
            <div className="h-2 overflow-hidden rounded bg-surface2"><div className="h-full bg-warn" style={{ width: `${progress}%` }} /></div>
            <div className="text-[10px] text-faint">{p2.available ? `gate evaluated ${ago(p2.age_seconds)}` : p2.error ?? "readiness report absent"}</div>
            {(p2.blocking_reasons ?? []).slice(0, 2).map((reason) => <div key={reason} className="truncate text-[10px] text-warn" title={reason}>{reason}</div>)}
          </div>
          <div className="mt-3 font-mono text-[10px] leading-relaxed text-faint">Live rows accrue continuously. This eligibility gate advances once per distinct valid weekday—not once per tick—so active capture and P2 blocked can both be true.</div>
        </div>
      </div>
      <div className="border-t p-3 hairline">
        <div className="eyebrow mb-2">Canonical domain inventory · signed manifests, snapshots, and partitions</div>
        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {activeDomains.length ? activeDomains.map((domain) => {
            const normalized = tier(domain, "normalized");
            const snapshots = tier(domain, "snapshots");
            const latest = [normalized, snapshots].filter(Boolean).sort((a, b) => (a?.freshness_age_seconds ?? Infinity) - (b?.freshness_age_seconds ?? Infinity))[0];
            return (
              <div key={domain.domain} className="rounded-md border p-3 hairline">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] font-semibold uppercase text-text">{domain.domain}</span>
                  <StatusDot color={latest ? POS : MUTE} />
                </div>
                <div className="space-y-1 font-mono text-[10.5px] text-faint">
                  <div className="flex justify-between"><span>normalized</span><span className="text-text">{normalized?.manifest_count ?? 0}</span></div>
                  <div className="flex justify-between"><span>snapshots</span><span className="text-text">{snapshots?.manifest_count ?? 0}</span></div>
                  <div className="flex justify-between"><span>partitions</span><span className="text-text">{(normalized?.partition_count ?? 0) + (snapshots?.partition_count ?? 0)}</span></div>
                  <div className="flex justify-between"><span>freshness</span><span className="text-text">{latest ? ago(latest.freshness_age_seconds) : "not captured"}</span></div>
                </div>
              </div>
            );
          }) : <NotConnected label="No canonical data domains found" />}
        </div>
      </div>
      {data.quality_failure_count > 0 && (
        <div className="border-t px-3 py-2 font-mono text-[10.5px] hairline">
          {data.quality_failures.slice(0, 8).map((failure) => (
            <div key={`${failure.kind}:${failure.path}`} className="flex gap-2 py-0.5"><span className="text-neg">{failure.kind}</span><span className="truncate text-faint" title={failure.path}>{failure.path}</span></div>
          ))}
          {data.quality_failures_truncated && <div className="text-warn">Additional failures omitted from the response.</div>}
        </div>
      )}
    </Card>
  );
}

function JobsPanel({ jobs }: { jobs: AxiomTrainingCockpit["jobs"] }) {
  return (
    <Card>
      <SectionTitle right={<span className="flex gap-2">{["submitted", "running", "failed", "completed"].map((status) => <Badge key={status} color={stateColor(status)}>{status.toUpperCase()} {jobs.counts[status] ?? 0}</Badge>)}</span>}>Jobs · signed manifest lineage</SectionTitle>
      <div className="divide-y hairline">
        {jobs.jobs.slice(0, 12).map((job) => (
          <div key={job.job_id} className="grid gap-2 px-3 py-2.5 font-mono text-[10.5px] md:grid-cols-[minmax(180px,1fr)_100px_minmax(180px,1fr)_160px]">
            <div className="min-w-0"><div className="truncate text-text" title={job.job_id}>{job.job_id}</div><div className="truncate text-faint">{job.lane ?? job.lanes?.join(", ") ?? "unknown lane"}</div></div>
            <Badge color={stateColor(job.status)} dot={job.status === "running"} pulse={job.status === "running"}>{job.status.toUpperCase()}</Badge>
            <div className="truncate text-faint" title={job.manifest_digest ?? undefined}>manifest {shortDigest(job.manifest_digest)}</div>
            <div className="text-right text-faint">{shortTime(job.completed_at ?? job.started_at ?? job.submitted_at)}</div>
            <div className="text-faint md:col-span-4">
              container <span className="text-text" title={job.container_digest ?? undefined}>{shortDigest(job.container_digest)}</span>
              {" · "}resource <span className="text-text">{job.resource_class ?? "not reported"}</span>
              {" · "}wall <span className="text-text">{job.resource_usage?.wall_seconds == null ? "not reported" : `${number(job.resource_usage.wall_seconds)}s`}</span>
              {" · "}cost <span className="text-text">{job.cost == null ? "not reported" : number(job.cost)}</span>
            </div>
            {job.error && <div className="text-neg md:col-span-4">{job.error}</div>}
          </div>
        ))}
        {!jobs.jobs.length && <NotConnected label="No submitted or observed evidence jobs" hint="Completed jobs appear automatically when immutable packages land." />}
      </div>
    </Card>
  );
}

function LaneRow({ lane }: { lane: EvidenceLaneView }) {
  const current = lane.current;
  const checks = current.gate_results ?? [];
  const passedChecks = checks.filter((check) => check.passed).length;
  return (
    <details className="border-t px-3 py-2.5 font-mono hairline first:border-t-0">
      <summary className="grid cursor-pointer list-none items-center gap-2 text-[10.5px] md:grid-cols-[minmax(220px,1.2fr)_130px_130px_120px_minmax(130px,0.8fr)]">
        <div className="min-w-0"><div className="truncate text-text">{lane.lane_id}</div><div className="text-faint">{lane.family} · {lane.package_count} package{lane.package_count === 1 ? "" : "s"}</div></div>
        <Badge color={stateColor(current.state)} dot>{current.state ?? "UNKNOWN"}</Badge>
        <span className={current.negative_result ? "text-neg" : "text-pos"}>{current.negative_result ? "NEGATIVE" : "ADMISSIBLE"}</span>
        <span className="text-faint">checks {passedChecks}/{checks.length}</span>
        <span className="truncate text-faint" title={current.package_digest}>{shortDigest(current.package_digest)}</span>
      </summary>
      <div className="mt-2 grid gap-2 rounded-md border p-2.5 hairline md:grid-cols-2">
        <div className="space-y-1 text-[10px] text-faint">
          <div>job manifest <span className="text-text" title={current.job_manifest_digest ?? undefined}>{shortDigest(current.job_manifest_digest)}</span></div>
          <div>disposition head <span className="text-text" title={current.disposition_head_digest ?? undefined}>{shortDigest(current.disposition_head_digest)}</span></div>
          <div>verification <span className={current.verification?.decision === "REJECT" ? "text-neg" : "text-text"}>{current.verification?.decision ?? "not persisted"}</span>{current.verification?.rejection_reason ? ` · ${current.verification.rejection_reason}` : ""}</div>
          <div>derived from <span className="text-text">{current.derived_from_package_digests?.length ? current.derived_from_package_digests.map((digest) => shortDigest(digest)).join(", ") : "none declared"}</span></div>
          <div>current champion <span className="text-text">{lane.champion ? shortDigest(String(lane.champion.package_digest ?? "")) : "none"}</span></div>
          <div>prior champion <span className="text-text">{lane.prior_champion ? shortDigest(lane.prior_champion.package_digest) : "none"}</span></div>
        </div>
        <div className="space-y-1 text-[10px]">
          {checks.length ? checks.map((check, index) => (
            <div key={`${check.check_id ?? "check"}-${index}`} className="flex gap-2">
              <span className={check.passed ? "text-pos" : "text-neg"}>{check.passed ? "PASS" : "FAIL"}</span>
              <span className="text-faint">{check.check_id ?? "unnamed check"}{check.details ? ` · ${check.details}` : ""}</span>
            </div>
          )) : <span className="text-faint">No persisted verification checks for this package.</span>}
        </div>
      </div>
    </details>
  );
}

function EvidencePanel({ evidence }: { evidence: AxiomTrainingCockpit["evidence"] }) {
  const disconnected = evidence.readers.filter((reader) => !reader.available);
  return (
    <Card>
      <SectionTitle right={<span className="flex gap-2"><Badge color={evidence.negative_result_count ? WARN : POS}>{evidence.negative_result_count} NEGATIVE</Badge><Badge color={POS}>{Object.keys(evidence.champions).length} CHAMPION</Badge><Badge color={disconnected.length ? WARN : POS}>{evidence.readers.length - disconnected.length}/{evidence.readers.length} READERS</Badge></span>}>Evidence · packages, disposition, verification, champions</SectionTitle>
      {evidence.lanes.map((lane) => <LaneRow key={lane.lane_id} lane={lane} />)}
      {!evidence.lanes.length && <NotConnected label="No evidence packages indexed" hint="Readers are connected; lanes appear when signed packages are imported." />}
      {disconnected.length > 0 && <div className="border-t px-3 py-2 font-mono text-[10px] text-warn hairline">Reader status: {disconnected.map((reader) => `${reader.family} (${reader.reason ?? "unavailable"})`).join(" · ")}</div>}
    </Card>
  );
}

function ForwardRow({ row }: { row: ForwardLaneMonitor }) {
  const shadowDays = row.shadow_age_seconds == null ? null : row.shadow_age_seconds / 86_400;
  return (
    <div className="grid gap-2 border-t px-3 py-2.5 font-mono text-[10.5px] hairline first:border-t-0 md:grid-cols-[minmax(200px,1.3fr)_90px_repeat(7,minmax(70px,0.55fr))]">
      <div className="min-w-0"><div className="truncate text-text">{row.lane_id}</div><div className="text-faint">{row.available ? `updated ${shortTime(row.updated_at)}` : "monitor not reporting"}</div></div>
      <Badge color={row.available ? POS : MUTE}>{row.status?.toUpperCase() ?? "NO DATA"}</Badge>
      <span title="shadow duration">{shadowDays == null ? "—" : `${shadowDays.toFixed(1)}d`}</span>
      <span title="forward Sharpe">S {number(row.forward_sharpe)}</span>
      <span title="drawdown">DD {number(row.drawdown)}</span>
      <span title="turnover">TO {number(row.turnover)}</span>
      <span title="cost">C {number(row.cost)}</span>
      <span title="exposure">E {number(row.exposure)}</span>
      <span title="deviation from promotion baseline">Δ {number(row.baseline_deviation)}</span>
      {(row.retirement_warnings ?? []).length > 0 && <div className="text-neg md:col-span-9">{row.retirement_warnings?.join(" · ")}</div>}
    </div>
  );
}

function ControlsPanel({ data, reload }: { data: AxiomTrainingCockpit; reload: () => void }) {
  const controls = data.controls;
  const [runCredential, setRunCredential] = useState("");
  const [promotionCredential, setPromotionCredential] = useState("");
  const [disposition, setDisposition] = useState("");
  const [busy, setBusy] = useState<"run" | "promote" | null>(null);
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const runCredentialObject = useMemo(() => parseObject(runCredential), [runCredential]);
  const promotionCredentialObject = useMemo(() => parseObject(promotionCredential), [promotionCredential]);
  const dispositionObject = useMemo(() => parseObject(disposition), [disposition]);
  const ready = controls.enabled && controls.operator_trust_configured;
  const observedActiveRun = data.jobs.jobs.find((job) =>
    job.lane === "risk_target" && (job.status === "submitted" || job.status === "running")
  );
  const observedPendingJob = pendingJobId
    ? data.jobs.jobs.find((job) => job.job_id === pendingJobId)
    : undefined;
  const pendingTerminal = observedPendingJob?.status === "completed" || observedPendingJob?.status === "failed";
  const optimisticJobId = pendingTerminal ? null : pendingJobId;
  const runLocked = Boolean(observedActiveRun || optimisticJobId);
  const runStatus = observedActiveRun?.status ?? (optimisticJobId ? observedPendingJob?.status ?? "submitted" : null);
  const runJobId = observedActiveRun?.job_id ?? optimisticJobId;

  async function run() {
    if (!runCredentialObject || !controls.run_request_template) return;
    if (!window.confirm("Train the risk-target volatility and drawdown models from the displayed dataset? The signed evidence result can only end at QUARANTINED or REJECTED.")) return;
    setBusy("run"); setMessage(null);
    const result = await evidenceTrainingAction("run", { credential: runCredentialObject, request: controls.run_request_template });
    const jobId = typeof result.data.job_id === "string" ? result.data.job_id : "accepted job";
    if (result.ok && typeof result.data.job_id === "string") setPendingJobId(result.data.job_id);
    setBusy(null);
    setMessage({ ok: result.ok, text: result.ok ? `Training submitted as ${jobId}. Job status and artifacts will update automatically.` : String(result.data.detail ?? result.data.error ?? "Run request failed") });
    reload();
  }

  async function promote() {
    if (!promotionCredentialObject || !dispositionObject) return;
    if (!window.confirm("Relay this exact operator-signed disposition event? Transition policy and signer authority will be re-verified.")) return;
    setBusy("promote"); setMessage(null);
    const result = await evidenceTrainingAction("promote", { credential: promotionCredentialObject, disposition: dispositionObject });
    setBusy(null);
    setMessage({ ok: result.ok, text: result.ok ? "Signed transition relayed and re-verified." : String(result.data.detail ?? result.data.error ?? "Transition relay failed") });
    reload();
  }

  return (
    <Card>
      <SectionTitle right={<span className="flex gap-2"><Badge color={controls.enabled ? POS : MUTE} dot>CONTROL {controls.enabled ? "ENABLED" : "DISABLED"}</Badge><Badge color={controls.operator_trust_configured ? POS : NEG}>OPERATOR TRUST {controls.operator_trust_configured ? "READY" : "ABSENT"}</Badge></span>}>Governed controls · request only</SectionTitle>
      <div className="grid gap-3 p-3 xl:grid-cols-2">
        <div className="rounded-md border p-3 hairline">
          <div className="eyebrow mb-2">Train risk-target model</div>
          <div className="mb-2 font-mono text-[10px] leading-relaxed text-faint">Fits the volatility and drawdown heads, evaluates their gates, signs the immutable artifacts, and queues them for review. Training never promotes a package or changes live sizing.</div>
          <div className="mb-2 font-mono text-[10px] text-faint">Subject {shortDigest(controls.run_subject_digest)} · dataset {shortDigest(controls.dataset_sha256)} · exact request:</div>
          <pre className="scroll-thin max-h-28 overflow-auto rounded bg-surface2 p-2 font-mono text-[10px] text-dim">{JSON.stringify(controls.run_request_template, null, 2)}</pre>
          <textarea value={runCredential} onChange={(event) => setRunCredential(event.target.value)} placeholder="Paste operator-signed run credential JSON" className="mt-2 h-32 w-full rounded-md border bg-transparent p-2 font-mono text-[10px] text-text outline-none hairline" />
          <button onClick={run} disabled={!ready || !controls.dataset_configured || !controls.run_request_template || !runCredentialObject || busy !== null || runLocked} className="mt-2 w-full rounded-md border px-3 py-2 font-mono text-[11px] font-semibold text-pos hairline disabled:cursor-not-allowed disabled:opacity-40">{busy === "run" ? "AUTHORIZING + QUEUING…" : runLocked ? `TRAINING ${runStatus?.toUpperCase()}` : "TRAIN RISK-TARGET MODEL"}</button>
          {runJobId && <div className="mt-2 truncate font-mono text-[10px] text-warn" title={runJobId}>Active job {runJobId}</div>}
        </div>
        <div className="rounded-md border p-3 hairline">
          <div className="eyebrow mb-2">Relay signed disposition transition</div>
          <div className="mb-2 font-mono text-[10px] text-faint">The dashboard cannot mint approval, skip transition policy, or write a champion pointer.</div>
          <textarea value={disposition} onChange={(event) => setDisposition(event.target.value)} placeholder="Paste operator-signed disposition envelope JSON" className="h-28 w-full rounded-md border bg-transparent p-2 font-mono text-[10px] text-text outline-none hairline" />
          <textarea value={promotionCredential} onChange={(event) => setPromotionCredential(event.target.value)} placeholder="Paste credential bound to the envelope payload digest" className="mt-2 h-28 w-full rounded-md border bg-transparent p-2 font-mono text-[10px] text-text outline-none hairline" />
          <button onClick={promote} disabled={!ready || !dispositionObject || !promotionCredentialObject || busy !== null} className="mt-2 w-full rounded-md border px-3 py-2 font-mono text-[11px] font-semibold text-warn hairline disabled:cursor-not-allowed disabled:opacity-40">{busy === "promote" ? "REVERIFYING…" : "RELAY SIGNED TRANSITION"}</button>
        </div>
      </div>
      <div className="border-t px-3 py-2 font-mono text-[10px] text-faint hairline">{controls.invariants.join(" · ")}</div>
      {message && <div className={`border-t px-3 py-2 font-mono text-[10.5px] hairline ${message.ok ? "text-pos" : "text-neg"}`}>{message.text}</div>}
    </Card>
  );
}

export function TrainingEvidenceCockpit() {
  const { data, error, loading, reload } = usePoll<AxiomTrainingCockpit>("/api/axiom_training", 5000);
  if (loading && !data) return <Card><SectionTitle>Training + Evidence</SectionTitle><Loading label="Loading evidence authority…" /></Card>;
  if (!data || !data.connected) return <Card><SectionTitle>Training + Evidence</SectionTitle><NotConnected label="Training cockpit unavailable" hint={data?.error ?? error ?? undefined} /></Card>;
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <DataPanel data={data.data} />
      <ControlsPanel data={data} reload={reload} />
      <JobsPanel jobs={data.jobs} />
      <EvidencePanel evidence={data.evidence} />
      <Card>
        <SectionTitle right={<Badge color={data.forward_monitoring.reporting_count ? POS : MUTE}>{data.forward_monitoring.reporting_count}/{data.forward_monitoring.total} REPORTING</Badge>}>Forward monitoring · factual telemetry only</SectionTitle>
        {data.forward_monitoring.lanes.map((row) => <ForwardRow key={row.lane_id} row={row} />)}
        {!data.forward_monitoring.lanes.length && <NotConnected label="No evidence lanes available for forward monitoring" />}
      </Card>
    </div>
  );
}
