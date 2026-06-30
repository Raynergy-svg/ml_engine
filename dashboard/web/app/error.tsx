"use client"; // Error boundaries must be Client Components (Next App Router).
import { useEffect } from "react";

// F-M5: route-level error boundary. Every panel renders untyped JSON from disk-backed
// endpoints; a single unexpected shape (contract drift) used to take down the ENTIRE
// cockpit (blank page). Now one bad render degrades to this honest card with a retry,
// and the rest of the operator's context (URL, the fact that data is live) is preserved.
export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface to the browser console for diagnosis (no telemetry sink in this app).
    console.error("AXIOM panel error:", error);
  }, [error]);

  return (
    <div className="mx-auto flex w-full max-w-[1640px] flex-col items-start gap-3 px-4 py-10">
      <div className="card w-full max-w-2xl p-5">
        <div className="eyebrow text-neg">cockpit render error</div>
        <h2 className="mt-1 font-mono text-[18px] font-semibold text-text">
          A panel failed to render
        </h2>
        <p className="mt-2 font-mono text-[12px] leading-relaxed text-dim">
          One component hit an unexpected data shape and was isolated so it could not
          blank the whole cockpit. The data layer and the bot are unaffected — this is a
          display-only fault.
        </p>
        {error?.message && (
          <pre className="mt-3 max-h-40 overflow-auto rounded-md border bg-surface2 p-2 font-mono text-[11px] text-faint hairline">
            {error.message}
            {error.digest ? `\n\ndigest: ${error.digest}` : ""}
          </pre>
        )}
        <div className="mt-4 flex gap-2">
          <button
            onClick={reset}
            className="rounded-md border px-3 py-1.5 font-mono text-[12px] text-cyan hairline hover:bg-surface2"
          >
            Retry
          </button>
          <button
            onClick={() => window.location.reload()}
            className="rounded-md border px-3 py-1.5 font-mono text-[12px] text-faint hairline hover:text-dim"
          >
            Reload cockpit
          </button>
        </div>
      </div>
    </div>
  );
}
