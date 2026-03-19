"""Minimal stdlib-only HTTP server for dashboard preview.
Serves the index.html template with no third-party dependencies.
For full API functionality, use run.py with conda environment.
"""
import http.server
import os
import sys
import json

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "src", "dashboard", "templates", "index.html")


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            with open(TEMPLATE, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith("/api/"):
            # Return stub JSON so the UI doesn't break
            stub = self._api_stub(self.path)
            payload = json.dumps(stub).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):
        self.do_GET()

    def _api_stub(self, path):
        if "/agents" in path:
            return [
                {"name": name, "enabled": True, "base_weight": 1.0,
                 "learned_weight": 1.0, "effective_weight": 1.0}
                for name in [
                    "trend", "mean_reversion", "volatility", "risk_sentinel",
                    "uncertainty", "execution_quality", "momentum", "news_risk",
                    "multi_timeframe", "pair_performance", "session_timing",
                    "support_resistance"
                ]
            ]
        if "/state" in path:
            return {"portfolio_snapshot": {"nav": 100000, "session_trades": 0,
                    "win_rate": 0, "total_realized_pnl": 0}}
        if "/learnings" in path:
            return {"content": "(Preview mode — run with conda for live data)"}
        if "/rules" in path:
            return {"content": "(Preview mode — run with conda for live data)"}
        if "/config" in path:
            return {"min_confidence": 0.55, "min_rr_ratio": 1.2,
                    "atr_sl_multiplier": 1.5, "atr_tp_multiplier": 2.0,
                    "use_correlation_filter": True, "max_portfolio_risk": 0.15}
        if "/prd" in path:
            return {"userStories": [
                {"id": "S1", "title": "RL Weight Learning", "priority": "P0", "passes": True},
                {"id": "S2", "title": "Trade Journal + Feedback Loop", "priority": "P0", "passes": True},
                {"id": "S3", "title": "Drawdown Guardian", "priority": "P0", "passes": True},
                {"id": "S4", "title": "Dashboard UI", "priority": "P1", "passes": False},
            ]}
        if "/scan/results" in path:
            return {"running": False, "results": []}
        return {}

    def log_message(self, format, *args):
        sys.stderr.write(f"[dashboard] {args[0]}\n")


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"Dashboard preview at http://127.0.0.1:{PORT}")
    server.serve_forever()
