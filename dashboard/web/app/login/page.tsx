"use client";
import { useState } from "react";
import { Logo } from "@/components/Logo";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        const params = new URLSearchParams(window.location.search);
        window.location.href = params.get("next") || "/";
        return;
      }
      if (res.status === 503) setError("Auth is not configured on the server.");
      else if (res.status === 401) setError("Invalid access key.");
      else setError(`Login failed (${res.status}).`);
    } catch {
      setError("Network error — is the server reachable?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="card w-full max-w-sm p-7">
        <div className="mb-6 flex justify-center">
          <Logo height={40} />
        </div>
        <div className="mb-1 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-faint">
          Secure terminal access
        </div>
        <p className="mb-6 text-center text-[12px] text-dim">
          Read-only view of the Buddy trading engine · OANDA fxPractice
        </p>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <input
            type="password"
            autoFocus
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Access key"
            className="w-full rounded-md border bg-surface2 px-3 py-2.5 font-mono text-[13px] text-text outline-none hairline focus:border-[var(--color-cyan)]"
          />
          <button
            type="submit"
            disabled={busy || !password}
            className="rounded-md py-2.5 font-mono text-[13px] font-semibold text-base disabled:opacity-40"
            style={{ background: "linear-gradient(120deg,var(--color-cyan),var(--color-emerald))" }}
          >
            {busy ? "Verifying…" : "Unlock AXIOM"}
          </button>
          {error && (
            <div className="rounded-md border px-3 py-2 text-center font-mono text-[11px]"
                 style={{ color: "#ff4d6d", borderColor: "#ff4d6d44" }}>
              {error}
            </div>
          )}
        </form>
        <p className="mt-6 text-center font-mono text-[10px] text-faint">
          single-user · session-cookied · no public exposure
        </p>
      </div>
    </div>
  );
}
