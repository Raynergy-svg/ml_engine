import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Pin the workspace root to this app — the repo has multiple lockfiles and
  // Next would otherwise infer the monorepo root and watch the wrong tree.
  turbopack: { root: path.resolve(__dirname) },
};

export default nextConfig;
