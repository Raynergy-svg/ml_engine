#!/usr/bin/env python3
"""Quick OANDA env + connectivity check before launching tick capture."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.oanda_practice import OandaPracticeClient, OandaApiError

def main():
    token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
    account = os.getenv("OANDA_ACCOUNT_ID")
    env = os.getenv("OANDA_ENVIRONMENT", "practice")

    print("=" * 50)
    print("OANDA Environment Diagnostic")
    print("=" * 50)

    if not token:
        print("❌ OANDA_API_TOKEN / OANDA_API_KEY not found in environment.")
        print("   Export it:  export OANDA_API_TOKEN='your-token'")
        print("   Or place it in .env:  echo OANDA_API_TOKEN=... \u003e .env")
        sys.exit(1)
    else:
        print(f"✅ Token present ({len(token)} chars, {env} env)")

    if not account:
        print("❌ OANDA_ACCOUNT_ID not found.")
        print("   Export it:  export OANDA_ACCOUNT_ID='001-001-...'")
        sys.exit(1)
    else:
        print(f"✅ Account ID present: {account}")

    print("\nTesting connectivity...")
    try:
        client = OandaPracticeClient()
        summary = client.get_account_summary()
        balance = summary.get("account", {}).get("balance", "N/A")
        currency = summary.get("account", {}).get("currency", "N/A")
        print(f"✅ Connected. Balance: {balance} {currency}")
        print("\n🚀 Ready to capture ticks!")
        print(f"   python scripts/run_tick_capture.py --pairs EUR_USD,GBP_USD,USD_JPY")
        sys.exit(0)
    except OandaApiError as e:
        print(f"❌ OANDA API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
