#!/usr/bin/env python3
"""Run the Buddy Trading Dashboard.

Usage: python3 src/dashboard/run.py [--port 8501]
"""
import os
import sys

# Set env to avoid SIP issues on macOS
os.environ.setdefault("UVICORN_LOOP", "asyncio")
os.environ.setdefault("UVICORN_HTTP", "h11")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8501
    uvicorn.run(
        "src.dashboard.app:app",
        host="127.0.0.1",
        port=port,
        loop="asyncio",
        http="h11",
    )
