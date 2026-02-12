#!/usr/bin/env python3
"""Test balance fetching for buddy command."""

from rich.console import Console
console = Console()

# Test the exact code path
equity = None
live_nav = None
trades_today = 0
max_trades_per_day = 30

if equity is None:
    try:
        from oanda_practice import OandaPracticeClient
        from datetime import datetime, timezone
        
        client = OandaPracticeClient.from_env()
        account_summary = client.get_account_summary()
        account = account_summary.get('account', {})
        live_nav = float(account.get('NAV', 0))
        
        # Count trades opened today
        today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        trades_result = client._request(
            'GET',
            f'/accounts/{client._config.account_id}/trades',
            params={'state': 'ALL', 'count': 100}
        )
        for t in trades_result.get('trades', []):
            if t.get('openTime', '').startswith(today_utc):
                trades_today += 1
        
        if live_nav > 0:
            equity = live_nav
            console.print(f'[green]💰 Live Balance: ${live_nav:,.2f}[/green] (fetched from OANDA)')
            trades_remaining = max_trades_per_day - trades_today
            if trades_remaining <= 5:
                console.print(f'[yellow]📊 Trades Today: {trades_today}/{max_trades_per_day} (⚠️ {trades_remaining} remaining)[/yellow]')
            else:
                console.print(f'[dim]📊 Trades Today: {trades_today}/{max_trades_per_day}[/dim]')
    except Exception as e:
        console.print(f'[red]ERROR: {e}[/red]')
        import traceback
        traceback.print_exc()

print(f'Final equity: ${equity:,.2f}')
