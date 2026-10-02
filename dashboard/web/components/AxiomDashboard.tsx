"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { type Overview, type Provenance, useOverview } from "@/lib/axiom2";
import "./axiom2.css";

const views = [
  "Overview",
  "Markets",
  "Evidence",
  "Portfolio",
  "Execution",
  "System",
] as const;
type View = (typeof views)[number];
const titles: Record<View, string> = {
  Overview: "The operating picture",
  Markets: "Market context",
  Evidence: "The evidence trail",
  Portfolio: "Portfolio & risk",
  Execution: "Execution & reconciliation",
  System: "System provenance",
};

function Icon({ kind, size = 24 }: { kind: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    Overview: (
      <>
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
      </>
    ),
    Markets: <path d="M4 20v-6m8 6V8m8 12V3" />,
    Evidence: (
      <>
        <path d="M5 3h9l5 5v13H5zM14 3v6h5M8 13h8M8 17h6" />
      </>
    ),
    Portfolio: (
      <>
        <path d="M12 3v9h9" />
        <circle cx="12" cy="12" r="9" />
      </>
    ),
    Execution: <path d="m14 2-10 12h7l-1 8 10-13h-7z" />,
    System: (
      <>
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v3m0 16v3M1 12h3m16 0h3" />
      </>
    ),
    Warning: (
      <>
        <path d="m12 3 10 18H2zM12 9v5" />
        <path d="M12 17h.01" />
      </>
    ),
    Arrow: <path d="M3 12h18m-6-6 6 6-6 6" />,
  };
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[kind] || paths.Evidence}
    </svg>
  );
}
function Panel({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`a2-panel ${className}`}>
      <h2>
        <span className="a2-square" />
        {title}
      </h2>
      {children}
    </section>
  );
}
function Fact({
  label,
  value = "Unavailable",
}: {
  label: string;
  value?: string;
}) {
  return (
    <div className="a2-fact">
      <span>{label}</span>
      <span>{value}</span>
    </div>
  );
}
function Source({ source }: { source: Provenance }) {
  return (
    <details className="a2-source">
      <summary>Inspect source · {source.status.toLowerCase()}</summary>
      <dl>
        <dt>Source</dt>
        <dd>{source.source}</dd>
        <dt>SHA256 of displayed bytes</dt>
        <dd>{source.digest || "Unavailable"}</dd>
        <dt>Source observation time</dt>
        <dd>{source.source_time || "Unknown"}</dd>
        <dt>Local file modification time (not source time)</dt>
        <dd>{source.local_file_modified_at || "Unavailable"}</dd>
        <dt>Authenticity</dt>
        <dd>Not verified by dashboard</dd>
      </dl>
    </details>
  );
}
function Market() {
  const [range, setRange] = useState("5D");
  return (
    <Panel title="MARKET CONTEXT" className="a2-market">
      <div
        className="a2-panel-tools"
        role="group"
        aria-label="Market time range"
      >
        {["1D", "5D", "1M", "3M"].map((r) => (
          <button
            key={r}
            aria-pressed={range === r}
            onClick={() => setRange(r)}
          >
            {r}
          </button>
        ))}
      </div>
      <div className="a2-market-label">
        <strong>US equities</strong>
        <span>
          NO OBSERVATIONS LOADED
          <br />
          <small>Session unknown · source coverage unknown</small>
        </span>
      </div>
      <div
        className="a2-chart-empty"
        role="img"
        aria-label={`No authenticated price observations available for ${range}. No illustrative prices are shown.`}
      >
        <Icon kind="Markets" size={38} />
        <strong>Market observations unavailable</strong>
        <span>No authenticated observation producer is bound.</span>
        <small>{range} window · price and volume remain unknown</small>
      </div>
      <div className="a2-market-meta">
        <span>
          Last trade <b>—</b>
        </span>
        <span>
          Bid / Ask <b>—</b>
        </span>
        <span>
          Source time <b>Unknown</b>
        </span>
      </div>
      <p className="a2-caption">
        Live observations have not been loaded. No sample price series.
      </p>
    </Panel>
  );
}
function Portfolio() {
  return (
    <Panel title="PORTFOLIO CONTEXT">
      <div className="a2-allocation">
        <div className="a2-ring">
          <span>
            Allocation
            <br />
            unavailable
          </span>
        </div>
        <div className="a2-legend">
          <Fact label="Observed holdings" value="Unknown" />
          <Fact label="Model targets" value="Unavailable" />
          <Fact label="Cash reserve" value="Unknown" />
        </div>
      </div>
      <Fact label="Account value" />
      <Fact label="Broker buying power" />
      <Fact label="Reported cash" />
      <Fact label="Settled cash" value="Unknown" />
      <p className="a2-caption">
        Broker observations are separate from model targets. No account is
        selected or exposed.
      </p>
    </Panel>
  );
}
function Evidence({ data }: { data: Overview | null }) {
  const rows = data?.evidence.rows || [];
  const [selected, setSelected] = useState<string | null>(null);
  const row = rows.find((r) => r.package_digest === selected);
  return (
    <Panel title="RESEARCH PIPELINE">
      <ol className="a2-pipeline">
        {["Registered", "Walk-forward", "Frozen candidate", "Promotion"].map(
          (s, i) => (
            <li key={s}>
              <span>{i + 1}</span>
              {s}
            </li>
          ),
        )}
      </ol>
      <p className="a2-caption">
        Stages describe the research contract. Progress is not inferred from a
        cache.
      </p>
      {rows.length ? (
        <>
          <label className="a2-select">
            Inspect evidence package
            <select
              value={selected || ""}
              onChange={(e) => setSelected(e.target.value || null)}
            >
              <option value="">Choose a package</option>
              {rows.map((r) => (
                <option key={r.package_digest} value={r.package_digest}>
                  {r.package_id} · {r.lane_id}
                </option>
              ))}
            </select>
          </label>
          <div className="a2-evidence-list">
            {rows.map((r) => (
              <button
                key={r.package_digest}
                onClick={() => setSelected(r.package_digest)}
                aria-pressed={selected === r.package_digest}
              >
                <strong>{r.package_id}</strong>
                <span>{r.lane_id}</span>
                <small>
                  Cached {r.cached_state || "unknown"} · authenticity unverified
                </small>
              </button>
            ))}
          </div>
          {row && (
            <dl className="a2-detail">
              <dt>Package digest</dt>
              <dd>{row.package_digest}</dd>
              <dt>Disposition head digest</dt>
              <dd>{row.head_event_digest || "Unavailable"}</dd>
              <dt>Evidence classification</dt>
              <dd>Unverified · no capital eligibility</dd>
            </dl>
          )}
        </>
      ) : (
        <div className="a2-empty">
          <Icon kind="Evidence" size={30} />
          <strong>
            {data?.evidence.provenance.status === "INVALID"
              ? "Evidence cache invalid"
              : "No equity evidence projection available"}
          </strong>
          <span>No experiment or successful stage is assumed.</span>
        </div>
      )}
      {data && <Source source={data.evidence.provenance} />}
    </Panel>
  );
}
function Attention({ navigate }: { navigate: (v: View) => void }) {
  return (
    <section className="a2-panel a2-attention">
      <h2>
        <Icon kind="Warning" />
        NEEDS ATTENTION
      </h2>
      {[
        {
          title: "Verify broker observations",
          detail: "Connection, account and source coverage are unverified.",
          view: "Markets",
        },
        {
          title: "Inspect genuine holdout evidence",
          detail: "Holdout proof is unverified. This surface cannot open it.",
          view: "Evidence",
        },
        {
          title: "Qualify live shadow + recovery",
          detail: "Twenty sessions and four cycles remain unqualified.",
          view: "Execution",
        },
      ].map((item, i) => (
        <button key={item.title} onClick={() => navigate(item.view as View)}>
          <span className="a2-number">0{i + 1}</span>
          <span>
            <strong>{item.title}</strong>
            <small>{item.detail}</small>
          </span>
          <Icon kind="Arrow" />
        </button>
      ))}
    </section>
  );
}
function Execution({ data }: { data: Overview | null }) {
  return (
    <Panel title="EXECUTION BOUNDARY">
      <div className="a2-denial">
        <Icon kind="Execution" size={42} />
        <div>
          <h3>No capital authority</h3>
          <p>
            Public execution is disabled. Offline fake transport tests do not
            qualify live execution.
          </p>
        </div>
      </div>
      <Fact label="Genuine holdout" value="Unverified" />
      <Fact label="Live shadow" value="Unqualified" />
      <Fact label="Reconciliation" value="Unverified" />
      <Fact label="Operational boundary" value="Unverified" />
      <div className="a2-disabled-actions">
        <button disabled>Review order</button>
        <button disabled>Submit order</button>
        <button disabled>Cancel order</button>
      </div>
      <p className="a2-caption">
        No financial actions are available here. Approval remains in the kernel
        and separately governed operator path.
      </p>
      <details className="a2-source">
        <summary>Inspect kernel denial prerequisites</summary>
        {data ? (
          <ul>
            {data.blocked_prerequisites.map((r) => (
              <li key={r}>{r.replaceAll("_", " ").toLowerCase()}</li>
            ))}
          </ul>
        ) : (
          <p>Kernel projection unavailable.</p>
        )}
      </details>
    </Panel>
  );
}
export default function AxiomDashboard() {
  const { data, loading, stale, error, reload } = useOverview();
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const signOut = async () => {
    try {
      const response = await fetch("/auth/logout", { method: "POST" });
      if (!response.ok) throw new Error("Unable to sign out.");
      window.location.assign("/login");
    } catch {
      setSignOutError("Unable to sign out. Retry.");
    }
  };
  const [view, setView] = useState<View>("Overview");
  useEffect(() => {
    const sync = () => {
      const name = window.location.hash.slice(1);
      if (name === "a2-content") return;
      setView(views.find((v) => v.toLowerCase() === name) || "Overview");
    };
    queueMicrotask(sync);
    window.addEventListener("hashchange", sync);
    window.addEventListener("popstate", sync);
    return () => {
      window.removeEventListener("hashchange", sync);
      window.removeEventListener("popstate", sync);
    };
  }, []);
  const navigate = (v: View) => {
    setView(v);
    window.history.pushState(null, "", `#${v.toLowerCase()}`);
  };
  return (
    <div className="a2-shell">
      <a
        className="a2-skip"
        href="#a2-content"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("a2-content")?.focus();
        }}
      >
        Skip to content
      </a>
      <aside className="a2-sidebar">
        <Link href="/" className="a2-logo" aria-label="Axiom 2.0 overview">
          AXIOM <span>2.0</span>
        </Link>
        <nav aria-label="Axiom workspace">
          {views.map((v) => (
            <button
              key={v}
              aria-current={view === v ? "page" : undefined}
              onClick={() => navigate(v)}
            >
              <Icon kind={v} />
              <span>{v}</span>
            </button>
          ))}
        </nav>
        <div className="a2-legacy">
          <Link href="/legacy">
            <Icon kind="Evidence" />
            Legacy FX
          </Link>
          <small>Separate OANDA / Tier7 workspace</small>
        </div>
      </aside>
      <div className="a2-workspace">
        <header className="a2-header">
          <span>
            WORKSPACE / <b>{view.toUpperCase()}</b>
          </span>
          <div>
            <span>READ-ONLY PROJECTION</span>
            <span className="a2-execution-disabled">⊘ Execution disabled</span>
            <button className="a2-signout" onClick={signOut}>
              Sign out
            </button>
          </div>
        </header>
        <main id="a2-content" tabIndex={-1}>
          <div className="a2-heading">
            <h1>{titles[view]}</h1>
            <p>
              {view === "Overview"
                ? "Markets, portfolio and the path to eligibility"
                : "Kernel facts, explicit gaps and inspectable provenance"}
            </p>
          </div>
          {signOutError && <p role="alert">{signOutError}</p>}
          <div className="a2-connection" role="status">
            <span>
              {loading
                ? "Loading projection…"
                : error
                  ? `Unavailable · ${error}${stale ? " Previous response retained as stale." : ""}`
                  : stale
                    ? "Stale projection · source freshness unknown"
                    : "Local projection received · source freshness unknown"}
            </span>
            <button onClick={reload}>Refresh</button>
            {error?.includes("Sign in") && <Link href="/login">Sign in</Link>}
          </div>
          <section className="a2-status" aria-label="Authority state">
            {[
              {
                kind: "Evidence",
                label: "EVIDENCE",
                value:
                  data?.evidence.provenance.status === "AVAILABLE"
                    ? "Cache available"
                    : "Unavailable",
                detail: "Authenticity unverified",
                tone: "",
              },
              {
                kind: "Warning",
                label: "PROMOTION",
                value: "Holdout unverified",
                detail: "Not capital-qualified",
                tone: "warning",
              },
              {
                kind: "Portfolio",
                label: "PORTFOLIO",
                value: "Inputs unverified",
                detail: "No live risk decision",
                tone: "",
              },
              {
                kind: "Execution",
                label: "EXECUTION",
                value: "No capital authority",
                detail: "Blocked by policy",
                tone: "danger",
              },
            ].map((s) => (
              <div key={s.label}>
                <Icon kind={s.kind} size={34} />
                <span>
                  <small>{s.label}</small>
                  <strong className={s.tone}>{s.value}</strong>
                  <p>{s.detail}</p>
                </span>
              </div>
            ))}
          </section>
          {view === "Overview" && (
            <>
              <div className="a2-grid a2-top">
                <Market />
                <Portfolio />
              </div>
              <div className="a2-grid a2-bottom">
                <Evidence data={data} />
                <Attention navigate={navigate} />
              </div>
            </>
          )}
          {view === "Markets" && (
            <>
              <Market />
              <Panel title="OBSERVATION CONTRACT">
                <Fact label="Broker source time" value="Unknown" />
                <Fact label="Local broker receipt time" />
                <Fact label="Session / halt coverage" value="Unknown" />
                <p className="a2-caption">
                  Quotes, bars and corporate actions require explicit source and
                  availability times. Local request time cannot replace source
                  time.
                </p>
              </Panel>
            </>
          )}
          {view === "Evidence" && (
            <>
              <Evidence data={data} />
              <Attention navigate={navigate} />
            </>
          )}
          {view === "Portfolio" && (
            <>
              <Portfolio />
              <Panel title="RISK INPUTS">
                <Fact
                  label="Account alias"
                  value="Unavailable · no account selected"
                />
                <Fact label="Risk decision" value="Unavailable" />
                <Fact
                  label="Positions / corporate actions"
                  value="Unverified"
                />
                <p className="a2-caption">
                  Reported cash, buying power and settled cash are distinct
                  facts. Missing settled funds block capital admission.
                </p>
              </Panel>
            </>
          )}
          {view === "Execution" && <Execution data={data} />}
          {view === "System" && (
            <Panel title="SOURCE & MODE">
              <Fact
                label="Projection contract"
                value={data?.schema_version || "Unavailable"}
              />
              <Fact label="Mode" value={data?.mode || "Unavailable"} />
              <Fact
                label="Local projection receipt"
                value={data?.received_at || "Unavailable"}
              />
              <Fact label="Source observation time" value="Unknown" />
              <p className="a2-caption">
                Byte digests identify files read by this service; they do not
                prove signatures, source freshness or qualification.
              </p>
              {data?.boundaries.map((b) => (
                <div className="a2-boundary" key={b.name}>
                  <h3>{b.name} boundary</h3>
                  <p>{b.profile || "Profile unavailable"}</p>
                  <Source source={b.provenance} />
                </div>
              ))}
            </Panel>
          )}
        </main>
        <footer className="a2-footer">
          <span>
            <Icon kind="System" size={18} />
            Broker connection: <b>Unknown</b>
          </span>
          <span>
            Quote freshness: <b>Unknown</b>
          </span>
          <span>
            Reconciliation: <b>Unverified</b>
          </span>
          <button onClick={() => navigate("Evidence")}>
            Inspect evidence <Icon kind="Arrow" size={18} />
          </button>
        </footer>
      </div>
    </div>
  );
}
