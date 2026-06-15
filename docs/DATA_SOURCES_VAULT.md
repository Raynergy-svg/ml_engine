# Data Sources Vault — Complete Collection Landscape

**Analyst:** Dex
**Date:** 2026-06-15
**Repo:** `ml_engine` / Buddy Scanner

---

## Current State (Ground Truth)

| Source | Status | Coverage | Quality | Integration |
|--------|--------|----------|---------|-------------|
| OANDA Practice | **Active** | 15 pairs, M15, ~5K bars (~2.5mo) | Retail spread, no depth | `src/data/harvest.py` → parquet |
| OANDA Order Book | **Active** | Position + Order snapshots, 26/ea | Summary buckets only | `src/data/order_book/persister.py` |
| IBKR TWS | **Dormant** | Full `reqHistoricalData` | Real tick optional | `src/brokers/ibkr.py` (stubs ready) |
| ForexFactory Calendar | **Active** | High-impact events | Human-readable | `src/data/news/source.py` |
| RSS Headlines | **Fallback** | VADER sentiment scoring | Noisy, sparse | `src/scanner/agents/news_calendar.py` |

**Total footprint:** ~1.9 MB parquet + ~650 KB JSONL order books. You're data-starved for a deep-learning model that expects years of history.

---

## Tier 1 — Retail / Free (Integrate This Week)

### 1.1 Dukascopy Bank — Free Tick-By-Tick
**What:** Swiss bank offering **free** historical tick data back to 1997 via ASCII/FTP.
**URL:** `https://www.dukascopy.com/swiss/english/marketwatch/historical/`
**Format:** Bi5 (compressed binary) → convert to CSV with JForex tools or `pandas.read_csv(..., compression='gzip')`
**Integration path:**
```python
# New file: src/data/sources/dukascopy_harvester.py
# Download bi5, decompress with lz4, parse to OHLCV + tick
# Dukascopy uses UTC+2 (GMT+2 DST) timestamps — convert to UTC
```
**Pairs:** 70+ FX crosses, plus CFDs (indices, metals, crypto).
**Resolution:** Tick, 1-second OHLC, M1, M15, H1, D1.
**Value prop:** Goes back to 1997. For EUR/USD alone that's ~135M ticks/year. **Gold for regime diversity.**

### 1.2 HistData.com — Free M1 ASCII
**What:** Curated free M1 data, ASCII format, cleaned gaps.
**URL:** `https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/`
**Format:** `YYYYMMDD HHMMSS;BID;ASK`
**Caveat:** Requires email registration. Data stops at 2023 for some pairs.
**Value prop:** Pre-cleaned, no weekend gaps, bid/ask explicitly split. **Critical for spread-sensitive backtests.**

### 1.3 TrueFX — Free Tick Archive (2012–2022)
**What:** Tick-by-tick trade data from Integral's ECN. Aggregated, anonymized.
**URL:** `https://www.truefx.com/` (discontinued live, archives still downloadable)
**Format:** CSV with millisecond timestamps, bid/ask per tick.
**Value prop:** **Actual ECN tick flow**, not synthetically sampled. Shows microstructure patterns retail brokers hide.
**Limitation:** Archives end ~2022. Still enormously useful for pre-training on microstructure.

### 1.4 Yahoo Finance (yfinance) — Equities / Crypto Macro
**What:** Already imported in `src/data/data_processing.py` but unused.
**Use case:** Train regime detectors on SPY, GLD, BTC-USD as **exogenous macro features**.
**Value prop:** Free, unlimited, goes back decades. FX doesn't move in isolation — ES + GC + CL correlations matter.

### 1.5 Polygon.io — Cheap Professional ($49–$199/mo)
**What:** REST + WebSocket APIs for stocks, FX, crypto.
**FX Coverage:** 900+ pairs, tick + aggregates.
**URL:** `https://polygon.io/`
**Cost:** $49/mo for Starter (includes FX aggregates), $199/mo for all ticks.
**Integration path:**
```python
# src/data/sources/polygon_harvester.py
import polygon.RESTclient as client
c = client("YOUR_API_KEY")
c.list_tickers(market="fx")  # all pairs
```
**Value prop:** Unified API for equities + FX + crypto. Real-time WebSocket. Clean JSON schema.

### 1.6 Twelve Data — Low-Cost Multi-Asset ($8–$49/mo)
**What:** FX, stocks, crypto, options. REST + WebSocket.
**URL:** `https://twelvedata.com/`
**Cost:** $8/mo for 8 API calls/sec, $49/mo for 80/sec.
**FX Coverage:** 1,100+ pairs, intraday back to 2019.
**Value prop:** Cheapest way to get "professional-ish" multi-asset data. REST schema is simpler than Polygon.

---

## Tier 2 — Commercial / Premium ($50–$500/mo)

### 2.1 Pepperstone Historical Data Feed
**What:** Pepperstone (Australian broker) provides historical tick data to clients.
**How:** Open a live account (min $200), email support for "tick data export".
**Format:** Tick or M1 CSV, bid/ask.
**Value prop:** Same venue as live trading — **zero distributional shift** if you eventually trade with them.

### 2.2 Darwinex (TradersTrust) — Tick Data API
**What:** Broker with an open-data ethos. Tick data via API for account holders.
**URL:** `https://www.darwinex.com/`
**Cost:** Free with live account.
**Value prop:** They also provide **strategy replication** data (crowdsourced alpha signals). Unique.

### 2.3 QuantConnect Data Library
**What:** Cloud backtesting platform with deep historical data.
**Cost:** Free tier for personal use, $8/mo for live trading.
**Data:** Tick, second, minute. Back to 1998 for FX.
**Value prop:** Not just data — it's **pre-loaded in a research environment**. You can prototype strategies on their cloud before downloading.

### 2.4 FXCM Historical Data Downloader
**What:** FXCM's native downloader for clients. M1 data back to 2002.
**URL:** `https://www.fxcm.com/markets/insights/historical-forex-data/`
**Cost:** Free with live/demo account.
**Format:** CSV, OHLC + volume. No tick data.
**Value prop:** Extremely clean, no gaps, weekend handling is precise.

---

## Tier 3 — Institutional / "Gold" Data (The Real Edge)

### 3.1 CME Market Data Platform (MDP 3.0) — Exchange-Native Feed
**What:** The **actual** Chicago Mercantile Exchange feed for FX futures (EUR/USD, GBP/USD, JPY/USD, etc.) and options.
**Access:** CME Group Market Data Services. Requires "Non-Display" license if algo-trading.
**Cost:** ~$500–$2,000/mo depending on redistribution rights. Non-display (personal use) is cheaper.
**Format:** FIX/FAST + SBE (Simple Binary Encoding). ITCH-like.
**Why it's gold:**
- **Sub-microsecond timestamps** (SBE format).
- **Order book depth** (10 levels+) for CME FX futures.
- **Trade volume by aggressor side** — you know if buyer or seller initiated.
- **Implied order book** from options for volatility surface.
**Integration path:**
```python
# Requires: pymdp3 or custom SBE decoder
# New file: src/data/sources/cme_mdp_decoder.py
# Decode multicast UDP → DataFrame with book_depth, aggressor_side, trade_condition
```
**Value prop:** This is where the big boys eat. If you want to train an agent on **actual exchange microstructure**, not a retail broker's thinned-down summary, this is it.

### 3.2 EBS Market (Electronic Broking Services)
**What:** Institutional interbank platform owned by NEX/ICAP. THE venue for EUR/USD spot.
**Access:** Requires Prime Broker relationship or data vendor subscription.
**Cost:** $10K+/mo direct. Through vendors: TickData.com, QuantGo.
**Format:** Millisecond tick, full order book (but anonymized).
**Why it's gold:**
- EBS prints are what **central banks watch** for fixing rates.
- Shows **real interbank flow**, not retail bucket-shop flow.
- Contains **"fixing spikes"** at 4pm London — a known exploitable pattern invisible to retail data.
**Alternative access:** TickData.com sells EBS tick history (one-time purchase, not subscription).

### 3.3 360T / Integral / Currenex — Wholesale ECN Flow
**What:** Institutional FX platforms with anonymous matching.
**Access:** Requires corporate entity + signed NDA.
**Cost:** Negotiable, typically $5K+/mo.
**What you get:**
- **Live streaming indicative prices** from 100+ banks.
- **Trade tape** (anonymized) showing large block trades.
- **Liquidity maps** by tenor.
**Why it matters:** If your model can detect **large block orders hitting the tape** before they're reflected in retail spreads, you front-run the retail move by 50–200ms.

### 3.4 CLS Bank Settlement Data
**What:** Continuous Linked Settlement — the global FX settlement utility.
**URL:** `https://www.cls-group.com/data-products/`
**Products:**
- **CLS Market Insight:** Hourly traded volume by pair.
- **CLS Hedging Impact:** Net hedge flow by currency.
**Cost:** ~$2,000–$5,000/mo.
**Why it's gold:**
- CLS settles **$6.6 trillion/day**. Their volume data is ground truth.
- **Hourly flow imbalances** predict intraday moves with surprising accuracy.
- Unique data: JPY hedging spikes before fiscal year-end, EUR repatriation flows.
**Integration:** CSV delivery hourly or daily. Easy to merge with your candle data.

### 3.5 QuantHouse (Refinitiv/QH) — FPGA-Normalized Feed
**What:** Hardware-normalized market data from 100+ venues, delivered via FPGA.
**Access:** Enterprise sales only.
**Cost:** $15K+/mo.
**Value prop:** They normalize **every major venue** into one schema. If you want to trade EUR/USD across CME, EBS, Reuters, MTF, and 3 retail brokers simultaneously, QuantHouse gives you a unified book.
**Realistic alternative:** Subscribe to **QuantHouse Symphony** (cloud API, cheaper, REST/WebSocket).

### 3.6 TickData.com — Historical Tick Archives (One-Time Purchase)
**What:** Not a streaming service — a **historical data vendor**.
**URL:** `https://www.tickdata.com/`
**Products:**
- Forex Tick Data: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD.
- Goes back to **1994** for majors.
- **Bid/ask/last** at millisecond resolution.
**Cost:** ~$300–$1,000 per pair for full history. One-time.
**Why it's the best ROI:**
- One-time cost, lifetime use.
- Cleaned, gap-filled, dividend/adjusted for CFDs.
- Includes **NBBO** (National Best Bid/Offer) snapshots.
- Data integrity guarantee (they'll fix bad ticks if you report them).

### 3.7 QuantGo — Alternative Data + EBS Access
**What:** Data marketplace for hedge funds.
**URL:** `https://www.quantgo.com/`
**Cost:** Subscription per dataset.
**Unique datasets:**
- EBS tick data (via partnership).
- Credit card transaction flow by country (predicts GDP → currency moves).
- Satellite imagery (retail parking lot counts → consumer spending → USD).
**Why it's gold:** **Cross-domain signal fusion.** Your SOTA raw-sequence model could ingest satellite + credit card + FX tick in one forward pass.

### 3.8 RavenPack / Accern / Bloomberg NEF — News Analytics
**What:** Machine-readable news sentiment with entity tagging.
**Bloomberg NEF (News Expression Feed):** Sub-10ms delivery of structured news events.
**Cost:** Bloomberg Terminal + $2,000/mo for NEF.
**Why it matters for Buddy:** Your `NewsRiskPolicy` currently uses time-of-day proxies. **RavenPack** or **Accern** would give it actual NLP-processed event scores.

---

## Tier 4 — Synthetic / Generative (The Frontier)

### 4.1 Agent-Based Market Simulation (Zero Hedge)
**What:** Generate infinite synthetic tick data with controlled regimes.
**Tools:**
- `stochastic-oasis` or custom Heston+jump models.
- Agent-based models: 10K zero-intelligence traders + 10 informed traders.
**Why:** Train your neural agents on **more market crashes than occurred in history**.
**Integration:** `src/data/synthetic_generator.py` — already have infra in `src/data/feature_engineering.py`.

### 4.2 GAN/Copula Market Generation
**What:** Train a GAN on real EUR/USD M15, then generate statistically indistinguishable synthetic years.
**Value:** Augments scarce regime data (e.g., only 3 major recessions in your 2.5mo window).
**Caution:** GANs memorize. Use copulas or flow-based models for better generalization.

### 4.3 NeurIPS Data Competitions
**What:** FX forecasting competitions release anonymized institutional-grade data.
**Recent:** Jane Street (Real-Time Market Data Forecasting) — Kaggle competition with actual anonymized market data.
**Value prop:** Jane Street data is **actual market-maker flow**. You can train on it legally.

---

## Recommended Acquisition Strategy

### Immediate (This Week) — $0
1. **Activate IBKR historical data** — `src/brokers/ibkr.py` already supports `reqHistoricalData`. Set `ibkr_host`, run backfill for 2 years M1. Zero cost if you have TWS open.
2. **Download Dukascopy** for EUR/USD, GBP/USD, USD/JPY tick data 2015–2025. That's ~10 years of tick. Write a bi5→parquet converter.
3. **YFinance macro proxies** — add SPY, GLD, UUP, VIX as exogenous features in your raw-sequence model's "known future inputs" channel.

### Short-Term (This Month) — $50–$300
4. **Polygon.io Starter** ($49/mo) — Replace OANDA as primary harvest source. Their aggregates are cleaner, no "practice account" throttling. Unified equities+FX API simplifies your `HybridInference` code.
5. **TickData.com one-time** ($500) — Buy 10-year EUR/USD tick archive. This is your **pre-training corpus** for the raw-sequence model.
6. **CLS Market Insight** (negotiate trial) — Merge hourly flow imbalance as a feature channel.

### Medium-Term (3 Months) — $2,000–$5,000/mo
7. **CME MDP 3.0 Non-Display** ($500–$1,000/mo) — Write SBE decoder, add CME FX futures order book as a new 5-channel input (bid_size, ask_size, last_size, aggressor_side, trade_condition).
8. **RavenPack news feed** ($2,000/mo) — Replace proxy-based `NewsRiskPolicy` with real event embeddings.
9. **EBS via TickData.com** (one-time $3,000) — 5 years of EBS EUR/USD tick. Train an "institutional flow" detector that flags when retail and interbank prices diverge.

---

## Data Architecture Recommendations

### Multi-Source Aggregation Layer
Your `HarvestScheduler` is single-source. Build a `MultiSourceHarvester`:
```python
class MultiSourceHarvester:
    def __init__(self, sources: List[DataSource]):
        self.sources = sources  # [OandaSource(), DukascopySource(), CmeSource()]
    
    def harvest(self, pair: str, start: datetime, end: datetime):
        # 1. Fetch from cheapest source first
        # 2. Cross-validate with second source (detect stale/glitched bars)
        # 3. Flag bars where sources disagree > 1 pip
        # 4. Persist "consensus" bars + "disputed" bars separately
```
This is how institutional shops handle data quality. OANDA alone has **stale bars during low liquidity** (Sunday open, holidays). Cross-validation catches it.

### Tick Data Storage
Your current parquet stores OHLCV. For tick data:
- **Parquet is still fine** — use `pyarrow` with partitioning by `year/month/day`.
- **Expected size:** EUR/USD tick, 10 years ≈ 80 GB compressed.
- **Fast queries:** Partition by pair + year. Load only the partition you need.
- **Index:** `(timestamp, pair, source)` for deduplication.

### Order Book Depth Upgrade
Your current OANDA order book is **summary buckets** (price bands with % of positions). Real institutional order books have:
- Individual price levels (not bands).
- Bid/ask depth at each level.
- Time-priority of resting orders.
- **Cancel/replace events** (shows intent).

If you get CME MDP data, build a proper `LimitOrderBook` class in `src/data/order_book/` and compute features like:
- Bid/ask slope (how steep is the book?).
- Order flow toxicity (VPIN). 
- Large-order detection (iceberg hunting).

---

## Closing Thought

You're building a Formula 1 car but fueling it with supermarket gasoline. OANDA practice data is fine for scalping on a $100K demo. For a state-of-the-art deep-learning ensemble that claims to beat the market, you need **institutional-grade kerosene**.

The good news: `src/data/harvest.py` is already architected for multi-source. The `DataLoader` class already handles validation and normalization. And your neural agents don't care where features come from — they just learn associations.

**Start with Dukascopy + IBKR this week (free, immediate).** Buy EUR/USD tick history from TickData.com next. Everything else is optimization.
