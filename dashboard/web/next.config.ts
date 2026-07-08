import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Pin the workspace root to this app — the repo has multiple lockfiles and
  // Next would otherwise infer the monorepo root and watch the wrong tree.
  turbopack: { root: path.resolve(__dirname) },
  // Next 16 blocks cross-origin requests to dev resources (blockCrossSiteDEV);
  // the HMR websocket at /_next/webpack-hmr was rejected because the dashboard is
  // served on 127.0.0.1 which wasn't allowlisted → ERR_INVALID_HTTP_RESPONSE →
  // no hot-reload → stale bundles persisted after edits. Allowlist the local dev
  // origins so HMR connects. (Add any other host/LAN-IP the dashboard is opened from.)
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
