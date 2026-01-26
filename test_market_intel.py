#!/usr/bin/env python
"""Test market intelligence components."""
from market_intelligence import MarketIntelligence, fetch_forex_news

# Test 1: Calendar
print('=== CALENDAR TEST ===')
intel = MarketIntelligence(enable_sentiment=False, enable_calendar=True, enable_online_learning=False)
if intel.calendar:
    events = intel.calendar.fetch_events('NZD_USD', hours_ahead=24)
    print(f'Events in next 24h: {len(events)}')
    for e in events[:5]:
        print(f'  - {e.event_name} ({e.impact}) @ {e.timestamp}')
else:
    print('Calendar not initialized!')

# Test 2: Sentiment
print('\n=== SENTIMENT TEST ===')
intel2 = MarketIntelligence(enable_sentiment=True, enable_calendar=False, enable_online_learning=False)
if intel2.sentiment:
    # Test with sample text
    result = intel2.sentiment.analyze('USD is rallying strongly against JPY today')
    print(f'Sample analysis: {result}')
    
    # Try fetching actual headlines
    try:
        headlines = fetch_forex_news('NZD_USD', max_articles=5)
        print(f'Fetched headlines: {len(headlines)}')
        for h in headlines[:3]:
            print(f'  - {h[:80]}...' if len(h) > 80 else f'  - {h}')
        
        if headlines:
            batch = intel2.sentiment.analyze_batch(headlines)
            print(f'Batch sentiment: {batch}')
    except Exception as e:
        print(f'News fetch error: {e}')
else:
    print('Sentiment not initialized!')

# Test 3: Online Learning
print('\n=== ONLINE LEARNING TEST ===')
intel3 = MarketIntelligence(enable_sentiment=False, enable_calendar=False, enable_online_learning=True)
if intel3.online_learner:
    stats = intel3.online_learner.get_performance_stats()
    print(f'Stats: {stats}')
    print(f'Trade buffer size: {len(intel3.online_learner.trade_buffer)}')
    if intel3.online_learner.trade_buffer:
        last = intel3.online_learner.trade_buffer[-1]
        print(f'Last trade: {last.trade_id} - {last.instrument} - PnL: {last.pnl_pips:.1f} pips')
else:
    print('Online learner not initialized!')

# Test 4: Full pre_trade_check
print('\n=== PRE-TRADE CHECK TEST ===')
intel_full = MarketIntelligence(enable_sentiment=True, enable_calendar=True, enable_online_learning=True)
can_trade, reason, data = intel_full.pre_trade_check('NZD_USD', headlines=['NZD surges on positive data', 'Risk appetite improves'])
print(f'Can trade: {can_trade}')
print(f'Reason: {reason}')
print(f'Data: {data}')
