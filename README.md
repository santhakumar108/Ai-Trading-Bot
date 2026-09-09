# AI-Assisted Trading Analysis & Paper Trading System

A conservative, evidence-based system that analyzes stocks across technical,
fundamental, news, and social-sentiment dimensions, combines them into a
calibrated confidence score, and trades **only** when multiple independent
signals agree and the reward clearly outweighs the risk after costs.

## This system does NOT

- Guarantee profits.
- Claim 99% (or any other near-certain) prediction accuracy.
- Trade just to be trading. **NO TRADE is the default and expected outcome
  most of the time.**
- Size positions based on "confidence" — position size comes only from
  account capital, risk-per-trade %, entry price, and stop-loss distance.
- Place real orders. The only broker implementation shipped is `PaperBroker`,
  which is physically incapable of sending a live order.

If you take away one idea from this README: a strategy with a 55% win rate
and a strong risk/reward ratio can be, and often is, better than one with an
80% win rate and occasional catastrophic losses. This system is built to
prefer the former and to say "NO TRADE" whenever it can't tell the
difference.

---

## 1. Architecture

```
trading_system/
├── config/          Central, validated configuration (weights, thresholds,
│                    risk limits). LIVE_TRADING_ENABLED defaults to False here.
│                    `nse_universe.py` (hand-maintained NIFTY 50/Next 50/
│                    sector-index snapshot) and `providers.py` (the ONE
│                    factory mapping `providers.*` config / env vars to
│                    concrete market-data/news/fundamentals/social adapters
│                    — see section 4).
├── data/            Market data engine (yfinance-backed OHLCV, volume,
│                    volatility, index/relative-strength context, retries,
│                    corporate-action fetch), `data_quality.py`
│                    (`DataQualityChecker`/`DataQualityReport` — spec
│                    section 3: every run gets an explicit OK/DEGRADED/
│                    INVALID status, below threshold ⇒ forced NO TRADE),
│                    `market_calendar.py` (NSE trading calendar, Asia/Kolkata
│                    timezone handling), and `point_in_time.py` (the
│                    `PointInTimeValue` primitive + look-ahead guard used
│                    throughout the point-in-time tests).
├── indicators/      Technical analysis: SMA/EMA/RSI/MACD/ATR/Bollinger/VWAP/OBV,
│                    combined into one explainable 0-100 technical score.
├── fundamentals/    Fundamental analysis with explicit data-quality/staleness
│                    tracking (never guesses a missing number).
├── news/            News analysis: source credibility, staleness, 5-class
│                    sentiment, contradiction detection, relevance/novelty/
│                    duplicate detection, confidence scoring.
├── sentiment/       Social-media sentiment: volume, acceleration, bot/spam
│                    and manipulation heuristics, explicit `data_available`
│                    flag (never assumed bullish/neutral when absent).
│                    Secondary signal only.
├── models/          Interpretable ML baseline (calibrated logistic regression)
│                    with leakage-safe features and time-series CV.
├── strategy/        The AI Signal Engine (multi-factor weighted scoring +
│                    confidence bands), the final Trade Filter approval gate,
│                    the structured trade-report builder (now including
│                    per-candidate Expected Value), and `pipeline.py` — the
│                    SINGLE shared decision path (`build_pipeline(config)`)
│                    used by BOTH paper trading and backtesting, gated first
│                    by data quality, so they can never silently diverge.
├── risk/            Per-trade risk/reward + position sizing + Expected
│                    Value (after transaction costs/slippage, at an assumed
│                    win probability — never conflated with "confidence"),
│                    account-level capital protection (daily/weekly loss
│                    limits, sector/position exposure caps), and the
│                    real-money readiness checklist.
├── backtesting/     Event-driven, no-look-ahead backtester that runs the
│                    EXACT SAME `strategy.pipeline.StrategyPipeline` as
│                    paper trading (not a separate technical-only stand-in),
│                    gated by a run-level data-quality check, with realistic
│                    costs, regime segmentation, extended performance
│                    metrics (CAGR/annualized volatility/Sharpe/Sortino/
│                    Calmar/recovery factor/monthly & yearly returns) and a
│                    NIFTY 50 buy-and-hold comparison (`metrics.py`,
│                    `backtester.py`'s `BenchmarkComparison`), a
│                    chronological TRAIN→VALIDATION→TEST walk-forward
│                    validator with regime-dependency flagging
│                    (`walk_forward.py`), a `StrategyHealthReport`
│                    overfitting/robustness verdict (`health_report.py`),
│                    and Monte Carlo bootstrap/resequencing robustness
│                    testing (`monte_carlo.py`) — plus
│                    `historical_providers.py` (point-in-time fundamentals/
│                    news/social/ML interfaces — honestly "unavailable" by
│                    default, never fabricated) and every execution
│                    assumption documented in plain language via
│                    `execution_assumptions()`.
├── paper_trading/   Wires every module together into one pipeline; places
│                    only simulated orders; maintains a full CSV trade
│                    journal (`journal.py`) AND a full decision log of every
│                    scan outcome, approved or rejected, with reasons
│                    (`decision_log.py`); tracks equity/drawdown history;
│                    `scanner.py`'s `UniverseScanner` scans the CONFIGURED
│                    universe (never a hardcoded list) in rate-limited
│                    batches with a short-TTL cache.
├── broker/          BrokerInterface abstraction + the only concrete
│                    implementation (PaperBroker). A future live connector
│                    plugs in here without touching strategy/risk code.
├── dashboard/       `common.py` (shared Market Regime / Data Quality / Top
│                    Candidates computations used by all three dashboards
│                    below), a Streamlit dashboard, a terminal (rich)
│                    dashboard, and a static HTML snapshot generator — all
│                    showing Market Regime, Data Quality, per-candidate
│                    Technical/Fundamental/News/Social/ML/Risk score and
│                    Expected Value, and paper Equity/Drawdown/Sharpe/Daily
│                    Risk Used. Always shows PAPER vs LIVE mode.
├── tests/           pytest suite for every module above, fully offline
│                    (synthetic data + dependency injection — no network
│                    calls required to run the tests).
├── logs/            Trade journal / decision log CSVs land here by default.
├── main.py          CLI: scan / paper-trade / backtest / walk-forward /
│                    dashboard / readiness.
└── requirements.txt
```

### Data flow for one trade decision

```
MarketDataProvider.get_daily_with_quality()  (live, OR a point-in-time slice during a backtest)
        │
        ▼
DataQualityChecker.check() → DataQualityReport  (spec section 3 — OK/DEGRADED/INVALID,
        │                                         computed once per run/scan; never skipped)
        ▼
TechnicalAnalyzer ──┐
FundamentalAnalyzer ─┤   (live: real analyzers / backtest: historical_providers.py,
NewsAnalyzer ────────┼──► SignalInputs      "unavailable" unless a real point-in-time
SocialSentimentAnalyzer┘                     provider is supplied — never fabricated)
        │
        ▼
StrategyPipeline.decide(inputs, ..., data_quality=report)   [strategy/pipeline.py]
        │
        ├─ Data-quality gate FIRST: below threshold ⇒ forced NO TRADE, every other
        │  signal skipped entirely (see `_data_quality_no_trade`) — regardless of
        │  what SignalEngine/TradeRiskCalculator/TradeFilter would otherwise say.
        ├─ SignalEngine (weighted score, renormalized across only AVAILABLE
        │                components, model-agreement + min-independent-
        │                signals gate, confidence bands)
        ├─ TradeRiskCalculator (entry/stop/target/RR/position size AND Expected
        │                       Value after costs at an assumed win probability —
        │                       sizing/EV computed from capital & stop, never
        │                       from confidence)
        ├─ CapitalProtection.pre_trade_check (daily/weekly loss, exposure limits)
        └─ TradeFilter (final approval gate — anything fails ⇒ NO TRADE)
        │
        ▼
build_trade_report (the structured, human-readable output incl. Expected Value)
        │
        ▼ (only if approved)
PaperBroker.place_order → TradeJournal + DecisionLog   [paper trading — every
        │                                                scan decision is logged,
        │                                                approved or not]
        (backtesting fills the SAME approved decision at the next bar's
         open instead, via Backtester.run() — see below)
```

`PaperTradingEngine.scan_symbol` (`paper_trading/engine.py`) and
`Backtester.run` (`backtesting/backtester.py`) both call
`StrategyPipeline.decide()` — never `SignalEngine`/`TradeRiskCalculator`/
`TradeFilter` directly — so a backtest exercises the identical multi-factor
decision logic that paper trading uses, not a simplified look-alike.
`tests/test_pipeline_parity.py` proves this and proves no future bar can
ever influence a past decision.

### Why the layers are separated this way

- **SignalEngine** decides *does this look good* (technical/fundamental/
  news/social/market/volume/volatility, weighted and confidence-scored).
- **TradeRiskCalculator** decides *how big, and at what price*, purely from
  capital and stop distance — never from confidence.
- **CapitalProtection** decides *is the account currently allowed to take on
  any new risk at all* (independent of any single trade's quality).
- **TradeFilter** is the last word: every checkbox in the spec must pass, or
  the final decision is forced to `NO TRADE`, no matter what the engine
  concluded.

This separation means a change to, say, position-sizing rules can never
accidentally leak into the signal-quality logic, and vice versa.

---

## 2. Required Python packages

See `requirements.txt`. Summary of what each is for:

| Package | Purpose |
|---|---|
| pandas, numpy | Core data handling and numerics |
| yfinance | Free market data (OHLCV, intraday, fundamentals) — no API key |
| PyYAML, python-dotenv | Configuration loading and `.env` secrets |
| feedparser | RSS-based free news source (Google News) |
| vaderSentiment | Lightweight, interpretable lexicon-based sentiment scoring |
| scikit-learn, joblib | Interpretable ML baseline (logistic regression), calibration |
| streamlit | Interactive dashboard |
| matplotlib | Plotting (equity curves, etc., if you extend the dashboard) |
| pytest, pytest-cov | Testing |
| rich, tabulate | Terminal dashboard / report formatting |

Install everything with:

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
```

---

## 3. Quick start

```bash
# 1. Copy env template (safe defaults; live trading stays disabled)
cp .env.example .env

# 2. Run the test suite (fully offline, no network required)
pytest

# 3. Scan a universe of symbols (uses free yfinance + Google News + Reddit)
python main.py scan AAPL MSFT

# 4. Run one paper-trading cycle (scans, executes approved trades, journals them)
python main.py paper-trade AAPL MSFT

# 5. Backtest one symbol
python main.py backtest AAPL --period 5y

# 6. Walk-forward validate one symbol (checks for overfitting)
python main.py walk-forward AAPL --period 5y

# 7. Dashboard
streamlit run dashboard/dashboard.py         # interactive
python main.py dashboard AAPL MSFT --html out.html   # static snapshot

# 8. See what's required before even considering live trading
python main.py readiness
```

You can run everything **without any API keys**. `--no-news` / `--no-social`
flags skip those network calls if you want a fully offline demo (they still
work with synthetic/injected data in the test suite regardless).

---

## 4. Configuration

All tunables live in `config/default_config.yaml` (see inline comments) and
can be overridden by a local `config.yaml` (same shape) or a handful of
environment variables for safety-critical values (`LIVE_TRADING_ENABLED`,
`EMERGENCY_STOP`, `PAPER_STARTING_CAPITAL`). Key defaults:

- Signal weights: Technical 25%, Market Condition 15%, Fundamentals 15%,
  News 15%, Social 10%, Volume/Price 10%, Risk/Volatility 10% (must sum to 1.0).
- Minimum confidence to trade: 70/100.
- Minimum risk/reward: 2.0 (prefer ≥2.5).
- Risk per trade: 0.75% of capital (0.5%–1% range).
- Max daily loss: 2%. Max weekly loss: 5%.
- Max simultaneous positions: 5. Max sector exposure: 30%. Max single
  position: 20% of capital.

Confidence bands: 0–59 NO TRADE · 60–69 WATCH · 70–79 MODERATE · 80–89 HIGH
CONFIDENCE · 90–100 VERY HIGH CONFIDENCE. **These are calibrated, explainable
weighted scores, not statistical probabilities of profit** — see the
calibration note below.

- **`min_independent_signals` (default 1):** at least this many of
  `{technical, fundamentals, news, social, ML}` must actually have real,
  available data behind them before a trade can be considered at all.
  Components with no data are excluded from scoring and their weight is
  proportionally redistributed across whatever IS available — never scored
  as a fabricated neutral 50 (see `strategy/signal_engine.py`). The default
  is deliberately **1**, not 2: with no real fundamentals/news/social/ML
  provider wired in (the honest default — see "Fundamentals / news / social
  sentiment during backtests" below), technical is the only component that
  can ever be available, so a default of 2+ would silently force NO TRADE
  on every single decision, forever, regardless of setup quality — a bug
  that was root-caused and fixed; see "Known issue, fixed: zero-trade
  backtests" below. Set this to 2 or higher explicitly once real
  fundamentals/news/social/ML data is actually available for the run, to
  require corroboration beyond technical alone.

### Known issue, fixed: zero-trade backtests caused by `min_independent_signals`

A 5-year backtest run against real NSE symbols (and the synthetic diagnostic
fixtures in `tests/test_zero_trade_regression.py` and
`tests/test_cross_stock_diagnostic.py`) previously produced **zero approved
trades for every symbol**, regardless of technical setup quality. Root
cause: the shipped default config set `decision_thresholds.min_independent_signals
= 2`, but a default `Backtester` (no fundamentals/news/social/ML provider
supplied) can only ever make `technical` available — `fundamentals`,
`news_sentiment`, and `social_sentiment` are always `None` via
`backtesting/historical_providers.py`'s `NoHistorical*Provider` classes, and
`ml_probability_up` is `None` unless an `HistoricalMLProvider` is explicitly
wired in. `available_independent_signals` was therefore always exactly `1`,
so `strategy/signal_engine.py`'s `available_independent_signals <
min_independent_signals` gate fired on **every bar of every symbol**,
unconditionally — before market regime, confidence, R:R, or expected value
were ever meaningfully exercised at scale. This was NOT "genuinely no valid
setups" and NOT a broken indicator/R:R/EV calculation; it was a single
config default that was structurally unsatisfiable given the system's own
honest data-availability defaults. Fix: the default was changed to `1` (see
above) — every other gate (confidence, model agreement, R:R ≥ 2:1,
expected value, liquidity/volatility bounds, data quality) is untouched and
exactly as strict as before.

### Benchmark & sector indices — NSE-first by default

This system's shipped default is **NSE/India-focused**, not generic:
`system.market` / `system.exchange` / `system.trading_calendar` all default
to `"NSE"`, `system.timezone` defaults to `"Asia/Kolkata"`, and
`config/default_config.yaml`'s `universe.symbols` ships as the full NIFTY 50
(see `config/nse_universe.py` — a hand-maintained snapshot, not
live-fetched; it will drift over time and should be refreshed periodically).
`universe.index_symbol` (used for market-trend, relative-strength scoring,
and the backtester's buy-and-hold benchmark comparison) defaults to
`"^NSEI"` explicitly in the shipped YAML.

`universe.index_symbol` is still **not** hard-coded in the underlying
resolution logic, though: leave it `null` in your own override and
`config/settings.py` fills it in from `system.market` (`"NSE"` → `^NSEI`
NIFTY 50, `"BSE"` → `^BSESN` SENSEX, `"US"` → `^GSPC` S&P 500) **only if you
haven't set it explicitly**; `market: "generic"` has **no preset** at
all — market trend, relative strength, and the benchmark comparison are all
honestly reported as `UNKNOWN`/unavailable rather than comparing, say,
NSE-listed stocks against the S&P 500. Override the whole block for a
different market or a custom universe:

```yaml
system:
  market: "NSE"
  exchange: "NSE"
  trading_calendar: "NSE"      # data/market_calendar.py: NSE fixed national
                                 # holidays + weekends; other names get
                                 # weekend-only calendars
  timezone: "Asia/Kolkata"
universe:
  index_symbol: "^NSEI"
  symbols: ["RELIANCE.NS", "TCS.NS", ...]
  sector_indices:
    IT: "^CNXIT"        # optional: sector name -> index, enables sector-trend
                          # scoring instead of "N/A" for symbols mapped to
                          # that sector via universe.sector_map
  scan_batch_size: 10             # paper_trading/scanner.py: symbols per batch
  scan_batch_delay_seconds: 1.0   # pause between batches (rate-limit handling)
  scan_cache_ttl_seconds: 60.0    # short-TTL cache for repeat scans
```

`tests/test_config.py` covers the NSE/BSE/US presets, the "generic has no
silent S&P 500 fallback" guarantee, explicit-override precedence, the
default-is-NSE-with-NIFTY-benchmark guarantee, and `sector_indices`
configurability. `tests/test_market_calendar.py` covers the trading
calendar; `tests/test_data_quality.py`, `tests/test_market_data.py`, and
`tests/test_failure_safe.py` cover the data-quality gate end to end.

### Data providers — configuration-based, never hard-coded credentials

`config.providers.{market_data,news,fundamentals,social}` (settable via
`MARKET_DATA_PROVIDER` / `NEWS_PROVIDER` / `FUNDAMENTALS_PROVIDER` /
`SOCIAL_PROVIDER` env vars — see `.env.example`) choose which adapter
`config/providers.py`'s factory functions construct; `"none"` is always an
honest no-op for every domain (reports everything unavailable, never
fabricates). No API key lives in this repo's source — a real paid-vendor
adapter reads its own credentials from the environment inside its own
`__init__`/fetch method. See `tests/test_providers_config.py`.

---

## 5. Testing

```bash
pytest                      # run everything
pytest tests/test_risk_engine.py -v     # one module
pytest --cov=. --cov-report=term-missing   # with coverage
```

Every module has its own test file under `tests/`. The whole suite runs
**offline**: market data, news, and social sources are all dependency-
injected with synthetic/fake data (see `tests/conftest.py`), so nothing in
CI depends on yfinance, Google News, or Reddit being reachable.

What's covered:
- `test_indicators.py` — indicator math, no-look-ahead property, bounded scores.
- `test_risk_engine.py` — the spec's own worked example (Entry 100/Stop 97/
  Target 108 → RR 2.67), illogical-stop rejection, capital protection state
  machine (daily/weekly loss halts, sector/position caps, emergency stop).
- `test_signal_engine.py` — NO-TRADE-by-default on insufficient history,
  model-disagreement rejection, confidence-threshold rejection, unavailable
  components excluded (not faked) from scoring, weight renormalization,
  and the `min_independent_signals` gate.
- `test_zero_trade_regression.py` — regression coverage for the
  zero-trade-by-default bug above: pins the default config's
  `min_independent_signals == 1`, and proves a genuine technical-only
  bullish setup CAN pass through the full `Backtester` (default config,
  no threshold overrides) end to end.
- `test_cross_stock_diagnostic.py` — multi-symbol diagnostic (spec-style):
  runs several independent synthetic symbols through the same pipeline,
  reports the stage-by-stage funnel (data → indicators → technical signal →
  risk/reward → EV → final trade) per symbol, and proves one symbol's data
  cannot contaminate another's decision — without requiring every symbol to
  produce a trade.
- `test_market_data_isolation.py` — proves two different ticker symbols
  fetched through the same `MarketDataProvider` never share the same
  underlying data (distinct objects, independent cache keys, mutating one
  never affects the other).
- `test_trade_filter.py` — every checklist item can independently force NO TRADE.
- `test_backtester.py` — runs the SAME `StrategyPipeline` as paper trading
  (not a technical-only stand-in), no-look-ahead entry timing (fills at the
  next bar's open), cost sensitivity, regime classification, metrics
  correctness, and that fundamentals/news/social are honestly reported
  `unavailable` (never fabricated) unless a real historical provider is
  supplied.
- `test_pipeline_parity.py` — proves paper trading and backtesting share
  identical decision logic (both via a literal shared `StrategyPipeline`
  instance, and via two independently-built pipelines from the same
  `Config`), and proves no future bar can ever change a past decision by
  running the real `Backtester.run()` loop on the same history truncated to
  two different lengths and diffing every recorded decision.
- `test_ml_baseline.py` — leakage-safe feature construction, time-series CV,
  drift detection.
- `test_news_analysis.py` / `test_sentiment.py` — single-source confidence
  cap, contradiction detection, bot/manipulation heuristics.
- `test_fundamentals.py` — stale/missing-data handling.
- `test_paper_trading.py` — broker fills/rejections, journal P&L math,
  end-to-end offline engine scan.
- `test_config.py`, `test_market_data.py` — config validation, safe defaults
  (including the default-is-NSE-with-NIFTY-benchmark guarantee), retry/
  backoff on transient fetch errors, `DataUnavailableError` never retried,
  timestamp dedup/timezone handling, corporate-action fetch honesty.
- `test_market_calendar.py` — NSE trading calendar (weekends, fixed national
  holidays only — never invented variable-date holidays), market-hours
  checks, `from_csv`.
- `test_point_in_time.py` — the `PointInTimeValue` primitive, including the
  spec's own worked example (a quarterly result released after a historical
  trade date must not be available before its release date).
- `test_data_quality.py` — every `DataQualityChecker` check individually
  (missing rows, duplicate timestamps, invalid OHLC, zero/abnormal volume,
  unexplained price jumps vs. explained corporate actions, staleness,
  holiday mismatches).
- `test_failure_safe.py` — the data-quality gate enforced end-to-end through
  `StrategyPipeline`/`Backtester`/`PaperTradingEngine`: an otherwise-bullish
  signal is still forced to NO TRADE on bad data quality, a corrupted
  backtest run is marked invalid with zero trades, a provider outage never
  crashes a scan or forces a trade.
- `test_providers_config.py` — the provider factory: `"none"` is an honest
  no-op for every domain, an unrecognized choice raises rather than silently
  falling back.
- `test_metrics_extended.py` — annualized volatility, Calmar ratio, recovery
  factor, monthly/yearly return buckets, `execution_assumptions()`, and the
  NIFTY buy-and-hold `BenchmarkComparison` (available/unavailable/invalidated
  cases).
- `test_walk_forward_segments.py` — TRAIN/VALIDATION/TEST are three
  genuinely separate, chronologically non-overlapping `BacktestResult`s;
  entries in each are gated to their own window; the regime-dependency
  check (flagged / not-enough-diversity / not-profitable-anywhere cases).
- `test_health_report.py` — every `StrategyHealthReport` verdict path
  (INSUFFICIENT DATA / HEALTHY / CAUTION / OVERFIT RISK) and the generic
  parameter-sensitivity harness (stable vs. sign-flip-unstable).
- `test_monte_carlo.py` — bootstrap vs. shuffle semantics, probability of
  ruin/net loss/worse-than-actual drawdown, percentile monotonicity, the
  "not a prediction" disclaimer is always present.
- `test_decision_log_and_scanner.py` — every scan decision (approved,
  rejected, or NO TRADE for lack of data) is logged with a reason;
  `UniverseScanner` defaults to `config.universe.symbols`, batches, caches
  within TTL, and isolates a single symbol's failure from the rest of the
  cycle.
- `test_report.py` — per-candidate Expected Value is populated from the risk
  assessment when one exists and is `None` (never fabricated) otherwise.
- `test_dashboard.py` — Market Regime / Data Quality summaries, Top
  Candidates ordering, and that both the CLI and HTML dashboards render
  PAPER/LIVE mode plus every spec section 18 field without crashing.

---

## 6. On "confidence" and calibration (please read)

The Signal Engine produces a deterministic, explainable **weighted score**,
built from transparent rules (see `strategy/signal_engine.py`). It is
**not** inherently a statistical probability of profit. Section 6 of the
original spec requires this score to be *calibrated* using historical
out-of-sample performance before being trusted operationally. In practice
that means: run `backtest` and `walk-forward` over a long, multi-regime
history, bucket historical trades by the confidence score they would have
received, and check whether, say, "80–89 HIGH CONFIDENCE" trades really did
win more often / lose less, historically and out-of-sample, than "60–69
WATCH" trades. Only after that exercise should the bands in
`config/default_config.yaml` be treated as meaningfully calibrated for your
specific universe and time period — and even then, past performance does
not guarantee future results.

**Important limitation, stated plainly**: `Backtester` runs the exact same
`StrategyPipeline` as paper trading — the architectural gap where
backtesting used a separate technical-only strategy has been closed (see
`strategy/pipeline.py`, `backtesting/historical_providers.py`,
`tests/test_pipeline_parity.py`). What has **not** changed is data
availability: free, point-in-time-correct historical news, fundamentals,
and social-sentiment datasets are not available in this environment, so by
default `Backtester` reports those three components as honestly
`unavailable` for every historical date (see `NoHistorical*Provider` in
`backtesting/historical_providers.py`) — excluded from scoring and their
weight redistributed, never fabricated as neutral or positive. A default
backtest run therefore measures technical + market-condition + volume +
volatility (+ ML, if you enable and properly train it out-of-sample via
`--ml` / `HistoricalMLProvider`) — the full multi-factor engine, honestly
scored on whatever data actually exists for that date. To also backtest the
fundamentals/news/social components, implement the corresponding
`Historical*Provider` interface against a real point-in-time dataset and
pass it into `Backtester(...)` — do not synthesize one.

**Two more limitations, stated just as plainly:**

- **Expected Value uses a conservative, non-calibrated default win
  probability (0.5).** `TradeRiskCalculator` computes EV after transaction
  costs/slippage/spread (spec section 10), but deliberately never derives
  "probability of winning" from the signal engine's confidence score — a
  weighted score is not a probability. `RiskConfig.default_win_probability`
  is a flat assumption; an optional `win_probability_fn(confidence)` hook
  exists for a genuinely calibrated table (e.g. built from walk-forward
  results), but nothing wires one in by default. Until you supply one,
  every EV number in a report or dashboard is only as good as that flat 50%
  assumption — treat it as a gate against obviously bad setups, not as a
  real profit estimate.
- **The data-quality gate is run-level, not per-bar, inside `Backtester`.**
  A single check runs once over the whole input series before simulation
  starts (see `data.data_quality.DataQualityChecker` and
  `execution_assumptions()`'s `data_quality_gate` entry); a series bad
  enough to fail it is never simulated at all, but a series that passes is
  not re-checked bar-by-bar for, say, one corrupted bar buried in the
  middle. `PaperTradingEngine.scan_symbol`, by contrast, DOES check quality
  fresh on every single scan (there's only ever "the current bar" to check
  there).

**This whole system was built and tested inside a network-sandboxed
environment** that can reach PyPI but not Yahoo Finance, Google News, or
NSE's own site. Every "real" data adapter (yfinance-backed market data and
fundamentals, Google-News-RSS-backed news, Reddit-backed social) is written
against real, documented, free APIs and is exercised in the test suite only
via dependency-injected fakes/mocks — **none of it has been live-network-
tested from within the environment that built it.** Test it against live
data yourself before trusting it operationally.

---

## 7. Real-money readiness

`python main.py readiness` prints the 14-item checklist from
`risk/readiness.py` (backtest, out-of-sample test, walk-forward test, paper
trading, transaction costs, slippage, drawdown understanding, risk limits,
emergency stop, API auth, order-failure handling, duplicate-order
protection, broker-connection-failure handling, market-data-failure
handling). Live trading should only ever be considered after a human has
deliberately attested to every item — this system does not, and should not,
flip `live_trading_enabled` on its own.

Even with `live_trading_enabled: true` in config, **the shipped code cannot
place a real order** — `PaperBroker` is the only `BrokerInterface`
implementation, and it always simulates. Adding a real broker means writing
a new `BrokerInterface` subclass (see `broker/broker_interface.py`) that
reads credentials from environment variables (never hard-coded) and checks
the live-trading flag and emergency-stop flag itself before submitting
anything.

---

## 8. Philosophy (spec section 21, restated)

Optimize for **positive expectancy + capital preservation**, not for the
number of trades, not for maximum historical profit, and not for win rate
alone. It should feel normal — even boring — for this system to scan ten
symbols and return ten `NO TRADE`s. That is the system working correctly,
not a bug.
