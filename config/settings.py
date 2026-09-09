"""
Central configuration for the trading system.

Design goals:
  * A single source of truth for every tunable number (weights, thresholds,
    risk limits) so nothing is hard-coded deep inside a module.
  * Safe by default: LIVE_TRADING_ENABLED starts False and can only become
    True via an explicit, auditable config change (never via code).
  * Layered overrides: default_config.yaml  ->  local config.yaml (optional)
    ->  environment variables (highest precedence, useful for secrets/CI).

Nothing in this module makes network calls or trading decisions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()  # loads a local .env file if present; never commit .env

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"

# Benchmark index used ONLY as a fallback when `universe.index_symbol` is not
# explicitly set by the user, keyed off `system.market`. There is
# deliberately no universal default here: an NSE-focused deployment that
# forgets to set an index would previously and silently fall back to the
# S&P 500 (^GSPC), which is the wrong benchmark for market-trend/relative-
# strength scoring on Indian equities. "generic" has NO preset -- market
# trend/relative-strength degrade explicitly to UNKNOWN rather than guessing.
BENCHMARK_PRESETS: Dict[str, str] = {
    "NSE": "^NSEI",   # NIFTY 50
    "BSE": "^BSESN",  # S&P BSE SENSEX
    "US": "^GSPC",    # S&P 500
}

# Heuristics used only to WARN (never to silently override) when a
# configured index_symbol looks mismatched with the configured market.
_MARKET_INDEX_HINTS: Dict[str, List[str]] = {
    "NSE": ["^NSEI", "^CNX", ".NS"],
    "BSE": ["^BSESN", ".BO"],
    "US": ["^GSPC", "^DJI", "^IXIC"],
}


@dataclass
class SystemConfig:
    live_trading_enabled: bool = False
    emergency_stop: bool = False
    base_currency: str = "INR"
    # "market" selects the BENCHMARK_PRESETS / _MARKET_INDEX_HINTS group
    # ("NSE", "BSE", "US", or "generic" for no preset). India/NSE is the
    # primary configuration for this system -- see README "NSE / Indian
    # market configuration".
    market: str = "NSE"
    # The specific listing venue -- kept as its own field (separate from
    # `market`) per the spec's explicit ask to separate universe / benchmark
    # / sector mapping / exchange / trading calendar / timezone. Affects
    # instrument-suffix conventions (".NS" for NSE, ".BO" for BSE) and, via
    # `trading_calendar`, which holiday rules apply. Typically equals
    # `market` for NSE/BSE/US deployments.
    exchange: str = "NSE"
    # Which data/market_calendar.NSECalendar-style calendar to use for
    # trading-day/holiday checks. Only "NSE" gets real fixed national
    # holidays baked in (see data/market_calendar.py); anything else is
    # honestly weekend-only rather than a guessed holiday list.
    trading_calendar: str = "NSE"
    timezone: str = "Asia/Kolkata"  # IST; never assume US market hours downstream


@dataclass
class UniverseConfig:
    # A short, safe default so constructing Config() directly (as most unit
    # tests do) doesn't implicitly try to scan 50 symbols. The full NIFTY 50
    # snapshot (see config/nse_universe.py) is what config/default_config.yaml
    # actually loads by default.
    symbols: List[str] = field(default_factory=lambda: ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
    # None means "not explicitly configured" -- load_config() resolves this
    # from BENCHMARK_PRESETS[system.market] if possible, and leaves it None
    # (explicit "unknown benchmark") for market="generic" rather than
    # guessing. Always set this explicitly for a real deployment; see
    # config/default_config.yaml and README "Benchmark & sector indices".
    index_symbol: Optional[str] = None
    sector_map: Dict[str, str] = field(default_factory=dict)
    # Optional: sector name -> benchmark index symbol (e.g. {"IT": "^CNXIT"}
    # for NSE IT-sector stocks), used for sector-trend scoring. A sector
    # with no entry here simply has no sector-trend component computed
    # (reported as "N/A", never fabricated).
    sector_indices: Dict[str, str] = field(default_factory=dict)
    # Scanner tuning (spec section 19: "don't scan thousands blindly if API
    # limits unreliable, use batching/caching/rate-limit handling") -- see
    # paper_trading/scanner.py's UniverseScanner, which reads these.
    scan_batch_size: int = 10
    scan_batch_delay_seconds: float = 1.0
    scan_cache_ttl_seconds: float = 60.0


@dataclass
class SignalWeights:
    technical: float = 0.25
    market_condition: float = 0.15
    fundamentals: float = 0.15
    news_sentiment: float = 0.15
    social_sentiment: float = 0.10
    volume_price_behavior: float = 0.10
    risk_volatility: float = 0.10

    def validate(self) -> None:
        total = sum(asdict(self).values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"signal_weights must sum to 1.0 (got {total:.4f}). "
                "Fix config/default_config.yaml or your override file."
            )


@dataclass
class ConfidenceBands:
    no_trade_max: int = 59
    watch_max: int = 69
    moderate_max: int = 79
    high_max: int = 89

    def label(self, score: float) -> str:
        if score < 0 or score > 100:
            raise ValueError(f"confidence score out of range 0-100: {score}")
        if score <= self.no_trade_max:
            return "NO TRADE"
        if score <= self.watch_max:
            return "WATCH"
        if score <= self.moderate_max:
            return "MODERATE"
        if score <= self.high_max:
            return "HIGH CONFIDENCE"
        return "VERY HIGH CONFIDENCE"


@dataclass
class DecisionThresholds:
    min_confidence_to_trade: float = 70.0
    min_risk_reward: float = 2.0
    preferred_risk_reward: float = 2.5
    max_model_disagreement: float = 0.35
    min_history_bars: int = 250
    # This is the floor on how many of {technical, fundamentals, news,
    # social, ML} must actually have real data behind them before a trade
    # can even be considered -- unavailable components never count toward
    # this (see strategy/signal_engine.py's docstring).
    #
    # DEFAULT IS 1, NOT 2 -- and this is a deliberate, load-bearing choice,
    # not a weakening of any risk control. Technical analysis, computed
    # directly from real (never fabricated) price/volume data, is a
    # legitimate signal on its own. Fundamentals/news/social are OPTIONAL
    # corroboration: `backtesting/historical_providers.py`'s NoHistorical*
    # providers report them as unavailable for EVERY date by default (no
    # free, point-in-time-correct historical dataset for them exists), and
    # even in live/paper trading they frequently fail their own staleness/
    # credibility/volume checks in `strategy/signal_engine.py`. A default of
    # 2 therefore does not mean "require one piece of corroboration" -- for
    # any run that doesn't have a real fundamentals/news/social/ML provider
    # wired up, `available_independent_signals` can structurally never
    # exceed 1 (technical), so `available_independent_signals <
    # min_independent_signals` fires on EVERY bar of EVERY symbol,
    # unconditionally, regardless of how good the technical setup is. That
    # is exactly the failure mode spec section 9 forbids: "do not reject
    # every trade solely because news/social/fundamentals are unavailable
    # unless that is an explicitly configured mandatory gate." (This was
    # discovered as the root cause of a widespread "0 approved trades across
    # every NSE symbol in a 5-year backtest" report -- see
    # tests/test_zero_trade_regression.py and README's "Known issue,
    # fixed" note.) All other gates -- min_confidence_to_trade, the model-
    # agreement/disagreement check, min_risk_reward, the expected-value gate,
    # liquidity/volatility bounds, and the data-quality gate -- are
    # completely unaffected by this value and remain exactly as strict as
    # configured. An operator who DOES have real fundamentals/news/social/ML
    # coverage and wants to require corroboration beyond technical alone can
    # still set this to 2 (or higher) explicitly in config.yaml -- several
    # tests (e.g. test_signal_engine.py's
    # test_min_independent_signals_gate_blocks_single_signal_trades) exercise
    # that stricter mode directly.
    min_independent_signals: int = 1
    # Minimum expected value per share, after costs, required to approve a
    # trade (spec section 10: "do not optimize only for win rate"). EV is
    # computed as assumed_win_probability * net_reward_per_share -
    # (1 - assumed_win_probability) * (risk_per_share + costs) -- see
    # risk/risk_engine.py. 0.0 means "must not have a negative expected
    # value at the assumed win rate"; raise it to demand a margin of safety.
    min_expected_value_per_share: float = 0.0
    # Spec Part 16: "if market regime is strongly bearish, raise the BUY
    # threshold." Added to min_confidence_to_trade ONLY when
    # SignalInputs.market_regime starts with "BEAR" (see
    # data/market_data.py's classify_regime()) -- never subtracted for any
    # other regime, so no regime can ever make trading EASIER, only harder.
    bearish_regime_confidence_bonus: float = 5.0
    # Spec Part 16 (news conflict override): a credible, high-confidence
    # news item that STRONGLY opposes an otherwise-approved direction can
    # veto it (NO TRADE), never the reverse. "Strong" = news_score at or
    # below this for a would-be BUY, or at/above (100 - this) for a
    # would-be SELL; "credible" = NewsAggregate.overall_confidence at or
    # above news_conflict_min_confidence. See strategy/signal_engine.py's
    # news-conflict gate.
    news_conflict_score_threshold: float = 20.0
    news_conflict_min_confidence: float = 50.0


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.0075
    max_daily_loss_pct: float = 0.02
    max_weekly_loss_pct: float = 0.05
    max_simultaneous_positions: int = 5
    max_sector_exposure_pct: float = 0.30
    max_single_position_pct: float = 0.20
    min_liquidity_avg_volume: float = 100_000
    max_atr_pct_of_price: float = 0.08
    min_atr_pct_of_price: float = 0.002
    transaction_cost_pct: float = 0.0010
    slippage_pct: float = 0.0007
    min_edge_after_costs_pct: float = 0.001
    # Conservative DEFAULT win probability used only for the expected-value
    # gate (decision_thresholds.min_expected_value_per_share) when no
    # calibrated win-rate-by-confidence-band function is supplied. 0.5 (a
    # coin flip) is deliberately not optimistic: this system does not claim
    # to know its own win rate until walk-forward validation demonstrates
    # one out-of-sample (see backtesting/walk_forward.py). A 2:1 theoretical
    # reward/risk ratio does NOT by itself imply a >50% chance of winning.
    default_win_probability: float = 0.5

    # -- Small-account mode (spec Part 10-14) -------------------------------
    # `risk_per_trade_pct` above (0.75%) is an institutional-scale default:
    # on a small account it makes `capital * risk_per_trade_pct` too tiny to
    # ever afford a single share of almost anything (e.g. 0.75% of INR 1,000
    # = INR 7.50). Rather than silently loosening risk_per_trade_pct itself
    # (which would also change large-account sizing), a SEPARATE, explicit,
    # pre-configured percentage applies uniformly whenever capital is below
    # `small_account_capital_threshold` -- decided BEFORE any specific
    # candidate is evaluated, never adjusted afterward to force a particular
    # trade through (spec Part 11 explicitly forbids that). See
    # risk/risk_engine.py's TradeRiskCalculator._effective_risk_pct().
    small_account_capital_threshold: float = 25_000.0
    small_account_risk_per_trade_pct: float = 0.02
    # -- Loss-streak capital protection (spec Part 26) -----------------------
    # After this many consecutive LOSING closed trades, CapitalProtection
    # (risk/risk_engine.py) reduces the effective risk-per-trade by
    # `risk_reduction_factor` for subsequent trades, until a win resets the
    # streak. This is a state-driven, transparent reduction -- it can only
    # ever REDUCE size, never increase it, and it never touches configured
    # thresholds/limits themselves.
    max_consecutive_losses_before_reduction: int = 3
    risk_reduction_factor: float = 0.5


@dataclass
class PaperTradingConfig:
    starting_capital: float = 1_000_000.0
    base_currency: str = "INR"
    poll_interval_seconds: int = 60


@dataclass
class NewsConfig:
    max_headline_age_hours: int = 48
    min_source_credibility: float = 0.4
    contradiction_window_hours: int = 24


@dataclass
class SocialConfig:
    min_mentions_for_signal: int = 25
    spam_bot_score_threshold: float = 0.6
    max_weight_single_post: float = 0.05


@dataclass
class BacktestConfig:
    brokerage_pct: float = 0.0010
    taxes_pct: float = 0.0005
    slippage_pct: float = 0.0007
    bid_ask_spread_pct: float = 0.0005
    initial_capital: float = 1_000_000.0


@dataclass
class DataQualityConfig:
    """Thresholds for data/data_quality.py's DataQualityChecker. A symbol/
    run whose quality_score falls below `min_quality_score_to_trade` is
    forced to NO TRADE (backtests are marked INVALID) regardless of what
    every other signal says -- see strategy/pipeline.py."""
    min_quality_score_to_trade: float = 0.70
    max_missing_row_fraction: float = 0.05       # of expected trading days in range
    max_stale_data_days: int = 5                 # last bar older than this -> stale
    abnormal_daily_return_threshold: float = 0.20  # single-day |return| above this is flagged
    min_volume_for_liquidity_check: float = 1.0    # zero/near-zero volume flagged


@dataclass
class MLConfig:
    """Spec Part 7 (ML trade-outcome meta-model). `enabled` is a config-level
    mirror of the CLI's `--ml` flag -- off by default so ML never silently
    activates. Consumed by `paper_trading/live_ml_provider.py`'s
    `LiveTradeOutcomeMLProvider`; see `models/ml_baseline.py`'s
    `build_trade_outcome_features` for what these parameters mean."""
    enabled: bool = False
    min_history_bars: int = 300
    min_auc: float = 0.53
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 3.0
    max_holding_days: int = 20
    n_splits: int = 5


@dataclass
class MacroConfig:
    """Spec Part 2: global + India macro context (US VIX/India VIX/crude/
    USD-INR). Unlike ML (config.ml, opt-in due to fit cost), this is core
    analysis -- `enabled=True` by default, always attempted, gracefully
    degrading to UNKNOWN per-series on a fetch failure (never fabricated,
    never crashes a scan). See data/macro_data.py."""
    enabled: bool = True
    cache_ttl_seconds: float = 300.0
    us_vix_symbol: str = "^VIX"
    india_vix_symbol: str = "^INDIAVIX"
    crude_symbol: str = "CL=F"
    usdinr_symbol: str = "INR=X"


@dataclass
class ProvidersConfig:
    """Which concrete adapter each data domain uses, resolved from
    MARKET_DATA_PROVIDER / NEWS_PROVIDER / FUNDAMENTALS_PROVIDER /
    SOCIAL_PROVIDER / MACRO_PROVIDER environment variables (see
    config/providers.py). Never put API keys here or in yaml -- read them
    from the environment inside the adapter that needs them."""
    market_data: str = "yfinance"      # "yfinance" | "none"
    news: str = "google_rss"           # "google_rss" | "none"
    fundamentals: str = "yfinance"     # "yfinance" | "none"
    social: str = "reddit"             # "reddit" | "none"
    macro: str = "yfinance"            # "yfinance" | "none"


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signal_weights: SignalWeights = field(default_factory=SignalWeights)
    confidence_bands: ConfidenceBands = field(default_factory=ConfidenceBands)
    decision_thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    social: SocialConfig = field(default_factory=SocialConfig)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)

    def validate(self) -> None:
        self.signal_weights.validate()
        if self.decision_thresholds.min_risk_reward < 1.0:
            raise ValueError("min_risk_reward must be >= 1.0")
        if not (0 < self.risk.risk_per_trade_pct <= 0.05):
            raise ValueError("risk_per_trade_pct looks unsafe (expected <= 5%)")
        # Small-account risk-per-trade gets a higher ceiling than the
        # large-account default (see RiskConfig docstring above) but is
        # still bounded -- this is a deliberately configured account-tier
        # setting, not an excuse to risk an unbounded fraction of a small
        # account on one trade.
        if not (0 < self.risk.small_account_risk_per_trade_pct <= 0.10):
            raise ValueError("small_account_risk_per_trade_pct looks unsafe (expected <= 10%)")
        if self.risk.small_account_capital_threshold < 0:
            raise ValueError("small_account_capital_threshold must be >= 0")
        if self.risk.max_consecutive_losses_before_reduction < 1:
            raise ValueError("max_consecutive_losses_before_reduction must be >= 1")
        if not (0 < self.risk.risk_reduction_factor <= 1.0):
            raise ValueError("risk_reduction_factor must be in (0, 1.0]")


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(raw: dict) -> dict:
    """
    A small set of high-value env-var overrides, mainly for safety switches
    and secrets-adjacent values that should never live in a committed yaml.
    """
    if os.getenv("LIVE_TRADING_ENABLED", "").lower() in ("1", "true", "yes"):
        raw.setdefault("system", {})["live_trading_enabled"] = True
    if os.getenv("EMERGENCY_STOP", "").lower() in ("1", "true", "yes"):
        raw.setdefault("system", {})["emergency_stop"] = True
    capital = os.getenv("PAPER_STARTING_CAPITAL")
    if capital:
        raw.setdefault("paper_trading", {})["starting_capital"] = float(capital)

    # Data provider selection (section 21): configuration-based, never a
    # hard-coded vendor choice, and never an API key -- those live in the
    # individual adapter classes' own env-var reads (see config/providers.py).
    providers_raw = raw.setdefault("providers", {})
    for env_var, key in [
        ("MARKET_DATA_PROVIDER", "market_data"),
        ("NEWS_PROVIDER", "news"),
        ("FUNDAMENTALS_PROVIDER", "fundamentals"),
        ("SOCIAL_PROVIDER", "social"),
        ("MACRO_PROVIDER", "macro"),
    ]:
        val = os.getenv(env_var)
        if val:
            providers_raw[key] = val.strip().lower()
    return raw


def _resolve_index_symbol(universe_raw: dict, market: str) -> dict:
    """
    Fills in `index_symbol` from BENCHMARK_PRESETS[market] ONLY when the
    user did not explicitly set one -- never silently overrides an explicit
    choice, and never guesses for market="generic" (or an unrecognized
    market string), which leaves index_symbol=None so downstream code
    treats market-trend/relative-strength as explicitly UNKNOWN instead of
    comparing against the wrong country's index.
    """
    universe_raw = dict(universe_raw)
    explicit = universe_raw.get("index_symbol")
    if explicit:
        hints = _MARKET_INDEX_HINTS.get(market, [])
        if hints and not any(h in explicit for h in hints):
            logger.warning(
                "config: universe.index_symbol=%r does not look like a %s benchmark "
                "(expected something like %s). If this is intentional, ignore this "
                "warning; if not, set universe.index_symbol explicitly in config.yaml.",
                explicit, market, hints[0],
            )
        return universe_raw

    preset = BENCHMARK_PRESETS.get(market)
    if preset:
        logger.info("config: universe.index_symbol not set; defaulting to %s for market=%r.", preset, market)
        universe_raw["index_symbol"] = preset
    else:
        logger.warning(
            "config: universe.index_symbol not set and market=%r has no benchmark preset -- "
            "market trend and relative strength will be reported as UNKNOWN until you set "
            "universe.index_symbol explicitly (e.g. '^NSEI' for NSE, '^GSPC' for US).",
            market,
        )
        universe_raw["index_symbol"] = None
    return universe_raw


def _dict_to_config(raw: dict) -> Config:
    system_raw = raw.get("system", {})
    universe_raw = _resolve_index_symbol(raw.get("universe", {}), system_raw.get("market", "generic"))
    cfg = Config(
        system=SystemConfig(**system_raw),
        universe=UniverseConfig(**universe_raw),
        signal_weights=SignalWeights(**raw.get("signal_weights", {})),
        confidence_bands=ConfidenceBands(**raw.get("confidence_bands", {})),
        decision_thresholds=DecisionThresholds(**raw.get("decision_thresholds", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        paper_trading=PaperTradingConfig(**raw.get("paper_trading", {})),
        news=NewsConfig(**raw.get("news", {})),
        social=SocialConfig(**raw.get("social", {})),
        backtesting=BacktestConfig(**raw.get("backtesting", {})),
        data_quality=DataQualityConfig(**raw.get("data_quality", {})),
        providers=ProvidersConfig(**raw.get("providers", {})),
        ml=MLConfig(**raw.get("ml", {})),
        macro=MacroConfig(**raw.get("macro", {})),
    )
    cfg.validate()
    return cfg


def load_config(override_path: Optional[str] = None) -> Config:
    """
    Load configuration with layered overrides:
      default_config.yaml  ->  override_path (or ./config.yaml if present)  ->  env vars

    Raises ValueError if the resulting config is internally inconsistent
    (e.g. weights don't sum to 1.0, unsafe risk-per-trade).
    """
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f) or {}

    candidate = override_path or os.getenv("TRADING_CONFIG_PATH") or "config.yaml"
    candidate_path = Path(candidate)
    if candidate_path.exists():
        with open(candidate_path, "r") as f:
            user_raw = yaml.safe_load(f) or {}
        raw = _merge(raw, user_raw)

    raw = _apply_env_overrides(raw)
    return _dict_to_config(raw)
