import pytest

from config.settings import Config, SignalWeights, load_config


# --- Benchmark/index resolution (spec: no hard-coded S&P 500 default) -----

def test_default_config_is_nse_focused_with_nifty_benchmark():
    """India/NSE is this system's primary configuration: the shipped
    default_config.yaml must resolve to the NIFTY 50 benchmark, IST
    timezone, and an NSE trading calendar -- never a US default."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.system.market == "NSE"
    assert config.system.exchange == "NSE"
    assert config.system.trading_calendar == "NSE"
    assert config.system.timezone == "Asia/Kolkata"
    assert config.universe.index_symbol == "^NSEI"
    assert len(config.universe.symbols) >= 40  # the NIFTY 50 snapshot, not a couple of demo tickers


def test_generic_market_leaves_index_symbol_unset_not_sp500(tmp_path):
    """market='generic' must NOT silently fall back to any benchmark --
    least of all the S&P 500, which would be the wrong index for an
    NSE/BSE-focused deployment that forgot to set one explicitly."""
    override = tmp_path / "config.yaml"
    override.write_text("system:\n  market: generic\nuniverse:\n  index_symbol: null\n")
    config = load_config(override_path=str(override))
    assert config.system.market == "generic"
    assert config.universe.index_symbol is None


def test_nse_market_preset_resolves_to_nifty(tmp_path):
    override = tmp_path / "config.yaml"
    override.write_text("system:\n  market: NSE\n")
    config = load_config(override_path=str(override))
    assert config.universe.index_symbol == "^NSEI"


def test_bse_market_preset_resolves_to_sensex(tmp_path):
    override = tmp_path / "config.yaml"
    override.write_text("system:\n  market: BSE\nuniverse:\n  index_symbol: null\n")
    config = load_config(override_path=str(override))
    assert config.universe.index_symbol == "^BSESN"


def test_us_market_preset_resolves_to_sp500_only_when_explicitly_us(tmp_path):
    override = tmp_path / "config.yaml"
    override.write_text("system:\n  market: US\nuniverse:\n  index_symbol: null\n")
    config = load_config(override_path=str(override))
    assert config.universe.index_symbol == "^GSPC"


def test_explicit_index_symbol_is_never_overridden_by_a_preset(tmp_path):
    override = tmp_path / "config.yaml"
    override.write_text('system:\n  market: NSE\nuniverse:\n  index_symbol: "^CUSTOMIDX"\n')
    config = load_config(override_path=str(override))
    assert config.universe.index_symbol == "^CUSTOMIDX"


def test_sector_indices_default_to_nse_presets_and_are_configurable(tmp_path):
    """The shipped default already wires common NSE sector indices (IT,
    BANK, ...); a bare dataclass Config() (used directly by most unit
    tests) still defaults to empty -- sector classification isn't
    auto-derived there. Both are independently configurable."""
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.universe.sector_indices.get("IT") == "^CNXIT"
    assert config.universe.sector_indices.get("BANK") == "^NSEBANK"

    assert Config().universe.sector_indices == {}

    override = tmp_path / "config.yaml"
    override.write_text('universe:\n  sector_indices:\n    IT: "^CUSTOMIT"\n')
    config2 = load_config(override_path=str(override))
    assert config2.universe.sector_indices["IT"] == "^CUSTOMIT"


def test_default_config_loads_and_validates():
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.system.live_trading_enabled is False
    config.validate()  # should not raise


def test_live_trading_defaults_false():
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.system.live_trading_enabled is False


def test_signal_weights_must_sum_to_one():
    bad = SignalWeights(technical=0.5, market_condition=0.5, fundamentals=0.5,
                         news_sentiment=0, social_sentiment=0, volume_price_behavior=0, risk_volatility=0)
    with pytest.raises(ValueError):
        bad.validate()


def test_unsafe_risk_per_trade_rejected():
    config = Config()
    config.risk.risk_per_trade_pct = 0.5  # 50% per trade -- clearly unsafe
    with pytest.raises(ValueError):
        config.validate()


def test_env_override_can_enable_live_trading(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    config = load_config(override_path="__nonexistent_override__.yaml")
    assert config.system.live_trading_enabled is True
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
