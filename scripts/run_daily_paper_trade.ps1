# Daily paper-trading run, scheduled via Windows Task Scheduler
# (weekdays, shortly after NSE market close). Scans the FULL NSE
# EQ-series universe (config\full_universe.yaml, ~1367 symbols --
# see config/nse_all_equity.py for the source/staleness caveat), not a
# small hand-picked list -- a multi-hour run, which is fine since it
# starts well after market close with nothing else competing for time.
# News + social sentiment + macro/global context are all fetched by
# default (no --no-news/--no-social passed) -- this is NOT limited to
# technical indicators alone. Appends this run's full output to
# logs\paper_trade_run_log.txt so results build up over time alongside
# logs\paper_trade_journal.csv / logs\decision_log.csv. The bot only
# opens a position when its own confidence/risk-reward/expected-value
# gates clear -- most days, most symbols, correctly produce NO TRADE;
# that is the designed default outcome, not a malfunction. Never places
# a real order (PaperBroker is paper-only by design).

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Santha Kumar\Downloads\trading_system"

$logPath = "logs\paper_trade_run_log.txt"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "`n===== Paper-trade run (full NSE universe): $timestamp ====="

& python main.py --config "config\full_universe.yaml" paper-trade 2>&1 | Add-Content -Path $logPath

Add-Content -Path $logPath -Value "===== End of run: $timestamp ====="
