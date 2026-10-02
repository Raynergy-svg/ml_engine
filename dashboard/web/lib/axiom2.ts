"use client";
import { useEffect, useState } from "react";

export type Provenance = {
  source: string;
  digest: string | null;
  source_time: string | null;
  local_file_modified_at: string | null;
  status: "AVAILABLE" | "UNAVAILABLE" | "INVALID";
  authenticity: string;
};
export type EvidenceRow = {
  lane_id: string;
  package_id: string;
  package_digest: string;
  cached_state: string | null;
  head_event_digest: string | null;
  evidence_class: string;
  capital_eligible: false;
};
export type Overview = {
  schema_version: "axiom.dashboard.v1";
  mode: string;
  received_at: string;
  source_time: null;
  provenance: string;
  execution_enabled: false;
  capital_authorized: false;
  evidence: { provenance: Provenance; rows: EvidenceRow[] };
  boundaries: {
    name: string;
    profile: string | null;
    provenance: Provenance;
    declared_execution_enabled: boolean | null;
  }[];
  market: {
    status: string;
    reason: string;
    source_time: null;
    received_at: null;
  };
  portfolio: {
    status: string;
    account_alias: null;
    account_value: null;
    reported_cash: null;
    buying_power: null;
    settled_cash: null;
    source_time: null;
    received_at: null;
  };
  qualification: {
    holdout: string;
    live_shadow: string;
    reconciliation: string;
    operational_boundary: string;
  };
  blocked_prerequisites: string[];
};

// Validate every consumed field. Unknown contracts fail before rendering.
export function parseOverview(value: unknown): Overview {
  const object = (v: unknown): v is Record<string, unknown> =>
    !!v && typeof v === "object" && !Array.isArray(v);
  const text = (v: unknown): v is string =>
    typeof v === "string" && v.length <= 4096;
  const nullableText = (v: unknown) => v === null || text(v);
  const digest = (v: unknown) =>
    v === null || (typeof v === "string" && /^[a-f0-9]{64}$/.test(v));
  const source = (v: unknown) =>
    object(v) &&
    text(v.source) &&
    digest(v.digest) &&
    nullableText(v.source_time) &&
    nullableText(v.local_file_modified_at) &&
    typeof v.status === "string" &&
    ["AVAILABLE", "UNAVAILABLE", "INVALID"].includes(v.status) &&
    text(v.authenticity);
  if (
    !object(value) ||
    value.schema_version !== "axiom.dashboard.v1" ||
    value.mode !== "READ_ONLY_LOCAL_PROJECTION" ||
    value.execution_enabled !== false ||
    value.capital_authorized !== false ||
    !text(value.received_at) ||
    !Number.isFinite(Date.parse(value.received_at)) ||
    value.source_time !== null ||
    !text(value.provenance)
  )
    throw new Error("Unsupported dashboard projection");
  const {
    evidence,
    boundaries,
    blocked_prerequisites,
    market,
    portfolio,
    qualification,
  } = value;
  if (
    !object(evidence) ||
    !source(evidence.provenance) ||
    !Array.isArray(evidence.rows) ||
    evidence.rows.length > 2000 ||
    !evidence.rows.every(
      (r) =>
        object(r) &&
        text(r.lane_id) &&
        text(r.package_id) &&
        typeof r.package_digest === "string" &&
        digest(r.package_digest) &&
        nullableText(r.cached_state) &&
        digest(r.head_event_digest) &&
        r.evidence_class === "UNVERIFIED" &&
        r.capital_eligible === false,
    ) ||
    !Array.isArray(boundaries) ||
    boundaries.length !== 3 ||
    !boundaries.every(
      (b) =>
        object(b) &&
        typeof b.name === "string" &&
        ["research", "portfolio", "execution"].includes(b.name) &&
        source(b.provenance) &&
        nullableText(b.profile) &&
        (b.declared_execution_enabled === null ||
          typeof b.declared_execution_enabled === "boolean"),
    ) ||
    !Array.isArray(blocked_prerequisites) ||
    blocked_prerequisites.length > 100 ||
    !blocked_prerequisites.every(text)
  )
    throw new Error("Invalid dashboard provenance");
  if (
    !object(market) ||
    market.status !== "UNAVAILABLE" ||
    !text(market.reason) ||
    market.source_time !== null ||
    market.received_at !== null ||
    !object(portfolio) ||
    portfolio.status !== "UNAVAILABLE" ||
    ![
      "account_alias",
      "account_value",
      "reported_cash",
      "buying_power",
      "settled_cash",
      "source_time",
      "received_at",
    ].every((k) => portfolio[k] === null) ||
    !object(qualification) ||
    qualification.holdout !== "UNVERIFIED" ||
    qualification.live_shadow !== "UNQUALIFIED" ||
    qualification.reconciliation !== "UNVERIFIED" ||
    qualification.operational_boundary !== "UNVERIFIED"
  )
    throw new Error("Invalid dashboard observation contract");
  return value as unknown as Overview;
}

export function useOverview() {
  const [state, setState] = useState<{
    data: Overview | null;
    error: string | null;
    loading: boolean;
    stale: boolean;
  }>({ data: null, error: null, loading: true, stale: false });
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    const run = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 12000);
      try {
        const response = await fetch("/api/axiom2/overview", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error(
            response.status === 401
              ? "Session expired. Sign in again."
              : "Projection service unavailable.",
          );
        const data = parseOverview(await response.json());
        if (!stopped)
          setState({
            data,
            error: null,
            loading: false,
            stale:
              Date.now() - Date.parse(data.received_at) > 30000 ||
              Date.parse(data.received_at) > Date.now() + 5000,
          });
      } catch (error) {
        if (!stopped)
          setState((previous) => ({
            ...previous,
            error:
              error instanceof Error
                ? error.message
                : "Projection unavailable.",
            loading: false,
            stale: !!previous.data,
          }));
      } finally {
        clearTimeout(timeout);
        if (!stopped) timer = setTimeout(run, 10000);
      }
    };
    void run();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [revision]);
  return { ...state, reload: () => setRevision((n) => n + 1) };
}
