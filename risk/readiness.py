"""
Real-Money Readiness Checklist (spec section 20).

A simple, explicit gate: live trading should never be turned on by editing
a config value alone. This module models the checklist as a set of named,
boolean items that a human must mark True only after actually performing
that step (running the backtest, reviewing walk-forward results, running
paper trading for a meaningful period, etc.) -- this code cannot verify the
steps were done well, only that someone has explicitly attested to each one.

`ReadinessChecklist.all_passed` is what main.py's `enable-live-trading`
command checks before it will even show the user how to flip
`live_trading_enabled` in config -- and even then, the flag change itself
must be a deliberate, manual, auditable edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

CHECKLIST_ITEMS: List[str] = [
    "backtest_completed",
    "out_of_sample_test_completed",
    "walk_forward_test_completed",
    "paper_trading_completed",
    "transaction_costs_included",
    "slippage_tested",
    "max_drawdown_understood",
    "risk_limits_tested",
    "emergency_stop_tested",
    "api_authentication_tested",
    "order_failure_handling_tested",
    "duplicate_order_protection_tested",
    "broker_connection_failure_handling_tested",
    "market_data_failure_handling_tested",
]

ITEM_DESCRIPTIONS: Dict[str, str] = {
    "backtest_completed": "Backtest completed across multiple market regimes.",
    "out_of_sample_test_completed": "Out-of-sample test (unseen data) completed.",
    "walk_forward_test_completed": "Walk-forward validation completed; no unresolved overfitting flags.",
    "paper_trading_completed": "A meaningful period of live-data paper trading completed and reviewed.",
    "transaction_costs_included": "Backtests and paper trading include realistic transaction costs.",
    "slippage_tested": "Slippage assumptions tested and deemed realistic.",
    "max_drawdown_understood": "Maximum historical drawdown is understood and personally acceptable.",
    "risk_limits_tested": "Daily/weekly loss limits and position sizing tested under simulated adverse conditions.",
    "emergency_stop_tested": "Emergency stop mechanism manually tested end-to-end.",
    "api_authentication_tested": "Live broker API authentication tested (in a safe/sandbox mode first).",
    "order_failure_handling_tested": "Order rejection / partial-fill handling tested.",
    "duplicate_order_protection_tested": "Duplicate-order protection tested (e.g. retried requests, network blips).",
    "broker_connection_failure_handling_tested": "Broker connection failure/reconnect handling tested.",
    "market_data_failure_handling_tested": "Market-data outage/staleness handling tested.",
}


@dataclass
class ReadinessChecklist:
    items: Dict[str, bool] = field(default_factory=lambda: {k: False for k in CHECKLIST_ITEMS})

    def mark(self, item: str, passed: bool = True) -> None:
        if item not in self.items:
            raise KeyError(f"Unknown checklist item '{item}'. Valid items: {list(self.items)}")
        self.items[item] = passed

    @property
    def all_passed(self) -> bool:
        return all(self.items.values())

    def pending_items(self) -> List[str]:
        return [k for k, v in self.items.items() if not v]

    def render(self) -> str:
        lines = ["Real-Money Readiness Checklist:"]
        for item in CHECKLIST_ITEMS:
            status = "[x]" if self.items[item] else "[ ]"
            lines.append(f"  {status} {ITEM_DESCRIPTIONS[item]}")
        if self.all_passed:
            lines.append("\nAll items complete. Live trading may be manually enabled by a human "
                          "who understands the risks -- this checklist does not enable it automatically.")
        else:
            lines.append(f"\n{len(self.pending_items())} item(s) remaining before live trading should be considered.")
        return "\n".join(lines)
