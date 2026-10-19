"""Trusted read-service composition. No connection registration or order route.

The host owns one explicitly configured adapter. Tool bindings/authentication
must be supplied by a separately authorized service environment; this code never
imports assistant tools, copies a connector session or creates credentials.
"""
from src.axiom2.brokers.robinhood_readonly import RobinhoodReadOnly

class TrustedReadHost:
    __slots__=('_adapter',)
    def __init__(self,adapter):
        if type(adapter) is not RobinhoodReadOnly:raise TypeError('EXACT_READ_ADAPTER_REQUIRED')
        self._adapter=adapter
    def capabilities(self):return self._adapter.capabilities()
    async def account_snapshot(self,account_alias):return await self._adapter.account_snapshot(account_alias)
    async def market_snapshot(self,instruments):return await self._adapter.market_snapshot(instruments)
    async def order_observations(self,account_alias):return await self._adapter.order_observations(account_alias)
